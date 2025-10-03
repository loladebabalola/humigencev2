# inference/router.py

import os
import sys
import time
import json
import yaml
import asyncio
import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import logging

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    user: Optional[str] = Field(None, description="User identifier for tenant routing")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class InferenceRouter:
    """FastAPI router for multi-tenant inference with load balancing"""
    
    def __init__(self, profile_path: str):
        self.profile_path = Path(profile_path)
        self.profile = self.load_profile()
        self.instances = self.profile.get('instances', [])
        self.router_config = self.profile.get('router', {})
        self.strategy = self.router_config.get('strategy', 'round_robin')
        self.entrypoint = self.router_config.get('entrypoint', 'localhost:8000')
        
        # Dynamic routing state
        self.current_instance = 0
        self.tenant_assignments = {}  # tenant_id -> instance_index
        self.instance_metrics = {}  # instance_index -> metrics
        self.tenant_manager = None  # Will be set dynamically
        
        # Backpressure control
        self.max_queue_size = 100
        self.request_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # HTTP client for forwarding requests
        self.http_client = None
        
        # Initialize metrics
        for i, instance in enumerate(self.instances):
            self.instance_metrics[i] = {
                'total_requests': 0,
                'active_requests': 0,
                'total_tokens': 0,
                'avg_latency': 0.0,
                'last_request_time': 0,
                'error_count': 0
            }
        
        # Start background tasks
        self.background_tasks = []
    
    def set_tenant_manager(self, tenant_manager):
        """Set tenant manager for dynamic tenant updates"""
        self.tenant_manager = tenant_manager
        
    def load_profile(self) -> Dict[str, Any]:
        """Load the inference profile configuration"""
        try:
            with open(self.profile_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]❌ Failed to load profile: {e}[/red]")
            sys.exit(1)
    
    async def startup(self):
        """Initialize the router"""
        console.print(f"[bold green]🌐 Starting Multi-Tenant Inference Router[/bold green]")
        console.print(f"📋 Profile: {self.profile_path.name}")
        console.print(f"🔀 Strategy: {self.strategy}")
        console.print(f"🎯 Instances: {len(self.instances)}")
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Start background tasks
        self.background_tasks.append(asyncio.create_task(self.health_monitor()))
        self.background_tasks.append(asyncio.create_task(self.metrics_collector()))
        
        console.print("[green]✅ Router initialized successfully![/green]")
    
    async def shutdown(self):
        """Cleanup resources"""
        console.print("\n🛑 [bold]Shutting down Multi-Tenant Inference Router[/bold]")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        console.print("[green]✅ Router shutdown complete[/green]")
    
    def select_instance(self, tenant_id: Optional[str] = None) -> int:
        """Select an instance based on the routing strategy"""
        if not self.instances:
            raise HTTPException(status_code=503, detail="No instances available")
        
        # Check tenant quota if tenant_manager is available
        if self.tenant_manager and tenant_id:
            tenant = self.tenant_manager.get_tenant(tenant_id)
            if tenant and tenant.quota:
                # Check if tenant has quota available
                is_valid, reason = self.tenant_manager.validate_tenant_request(tenant_id, 0)
                if not is_valid:
                    raise HTTPException(status_code=429, detail=f"Tenant quota exceeded: {reason}")
        
        if self.strategy == "round_robin":
            instance_idx = self.current_instance
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance_idx
        
        elif self.strategy == "tenant_sticky":
            if tenant_id and tenant_id in self.tenant_assignments:
                return self.tenant_assignments[tenant_id]
            else:
                # Assign to least loaded instance
                instance_idx = min(range(len(self.instances)), 
                                 key=lambda i: self.instance_metrics[i]['active_requests'])
                if tenant_id:
                    self.tenant_assignments[tenant_id] = instance_idx
                return instance_idx
        
        elif self.strategy == "tenant_sticky_then_latency":
            if tenant_id and tenant_id in self.tenant_assignments:
                assigned_idx = self.tenant_assignments[tenant_id]
                # Check if assigned instance is healthy
                if self.is_instance_healthy(assigned_idx):
                    return assigned_idx
                else:
                    # Fall back to latency-based selection
                    del self.tenant_assignments[tenant_id]
            
            # Select instance with lowest average latency
            instance_idx = min(range(len(self.instances)), 
                             key=lambda i: self.instance_metrics[i]['avg_latency'])
            if tenant_id:
                self.tenant_assignments[tenant_id] = instance_idx
            return instance_idx
        
        else:
            # Default to round robin
            instance_idx = self.current_instance
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance_idx
    
    def is_instance_healthy(self, instance_idx: int) -> bool:
        """Check if an instance is healthy"""
        if instance_idx >= len(self.instances):
            return False
        
        metrics = self.instance_metrics.get(instance_idx, {})
        error_rate = metrics.get('error_count', 0) / max(metrics.get('total_requests', 1), 1)
        
        # Consider instance unhealthy if error rate > 50%
        return error_rate < 0.5
    
    async def forward_request(self, request: ChatCompletionRequest, instance_idx: int) -> Dict[str, Any]:
        """Forward request to a specific instance"""
        instance = self.instances[instance_idx]
        port = instance['port']
        
        # Update metrics
        self.instance_metrics[instance_idx]['total_requests'] += 1
        self.instance_metrics[instance_idx]['active_requests'] += 1
        self.instance_metrics[instance_idx]['last_request_time'] = time.time()
        
        start_time = time.time()
        
        try:
            # Prepare request for llama-server
            llama_request = {
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "max_tokens": request.max_tokens or 512,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream
            }
            
            # Forward to llama-server
            url = f"http://localhost:{port}/v1/chat/completions"
            response = await self.http_client.post(url, json=llama_request)
            response.raise_for_status()
            
            result = response.json()
            
            # Update metrics
            latency = time.time() - start_time
            self.update_instance_metrics(instance_idx, latency, result.get('usage', {}).get('total_tokens', 0))
            
            return result
            
        except httpx.TimeoutException:
            self.instance_metrics[instance_idx]['error_count'] += 1
            raise HTTPException(status_code=504, detail="Request timeout")
        except httpx.HTTPStatusError as e:
            self.instance_metrics[instance_idx]['error_count'] += 1
            raise HTTPException(status_code=e.response.status_code, detail=f"Upstream error: {e.response.text}")
        except Exception as e:
            self.instance_metrics[instance_idx]['error_count'] += 1
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        finally:
            self.instance_metrics[instance_idx]['active_requests'] -= 1
    
    def update_instance_metrics(self, instance_idx: int, latency: float, tokens: int):
        """Update metrics for an instance"""
        metrics = self.instance_metrics[instance_idx]
        
        # Update average latency (exponential moving average)
        alpha = 0.1
        if metrics['avg_latency'] == 0:
            metrics['avg_latency'] = latency
        else:
            metrics['avg_latency'] = alpha * latency + (1 - alpha) * metrics['avg_latency']
        
        metrics['total_tokens'] += tokens
    
    async def health_monitor(self):
        """Background task to monitor instance health"""
        while True:
            try:
                for i, instance in enumerate(self.instances):
                    port = instance['port']
                    
                    try:
                        # Simple health check
                        response = await self.http_client.get(f"http://localhost:{port}/health", timeout=5.0)
                        if response.status_code != 200:
                            logger.warning(f"Instance {i} (port {port}) health check failed")
                    except:
                        logger.warning(f"Instance {i} (port {port}) is unreachable")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def metrics_collector(self):
        """Background task to collect and log metrics"""
        while True:
            try:
                # Log metrics every minute
                await asyncio.sleep(60)
                
                logger.info("Instance metrics:", extra={
                    "metrics": self.instance_metrics,
                    "tenant_assignments": len(self.tenant_assignments)
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")

# FastAPI app
app = FastAPI(
    title="Multi-Tenant Inference Router",
    description="OpenAI-compatible API for multi-tenant inference",
    version="1.0.0"
)

# Global router instance
router_instance = None

@app.on_event("startup")
async def startup_event():
    global router_instance
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args, _ = parser.parse_known_args()
    
    router_instance = InferenceRouter(args.profile)
    await router_instance.startup()

@app.on_event("shutdown")
async def shutdown_event():
    global router_instance
    if router_instance:
        await router_instance.shutdown()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    healthy_instances = sum(1 for i in range(len(router_instance.instances)) 
                          if router_instance.is_instance_healthy(i))
    
    return {
        "status": "healthy" if healthy_instances > 0 else "unhealthy",
        "instances": len(router_instance.instances),
        "healthy_instances": healthy_instances,
        "strategy": router_instance.strategy
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    return {
        "instances": router_instance.instance_metrics,
        "tenant_assignments": router_instance.tenant_assignments,
        "strategy": router_instance.strategy
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if not router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    # Check queue capacity
    if router_instance.request_queue.qsize() >= router_instance.max_queue_size:
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    tenant_id = request.user
    
    # Select instance
    try:
        instance_idx = router_instance.select_instance(tenant_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    # Forward request
    try:
        result = await router_instance.forward_request(request, instance_idx)
        
        # Add request ID to response
        result['id'] = request_id
        
        # Record request for tenant tracking
        if router_instance.tenant_manager and tenant_id:
            tokens = result.get('usage', {}).get('total_tokens', 0)
            router_instance.tenant_manager.record_request(tenant_id, tokens, success=True)
        
        # Log request
        logger.info("Request processed", extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "instance_idx": instance_idx,
            "instance_port": router_instance.instances[instance_idx]['port'],
            "tokens": result.get('usage', {}).get('total_tokens', 0)
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request processing error: {e}", extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "instance_idx": instance_idx
        })
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Tenant Inference Router",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Tenant Inference Router")
    parser.add_argument("--profile", required=True, help="Path to profile YAML file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Store profile path for startup event
    sys.argv = ["router.py", "--profile", args.profile]
    
    console.print(f"[bold green]🌐 Starting Multi-Tenant Inference Router[/bold green]")
    console.print(f"📋 Profile: {args.profile}")
    console.print(f"🌐 Host: {args.host}:{args.port}")
    
    uvicorn.run(
        "router:app",
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
