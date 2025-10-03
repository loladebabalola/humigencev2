# inference/tenant_manager.py

import os
import json
import time
import secrets
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
import typer

console = Console()

@dataclass
class TenantQuota:
    """Tenant quota configuration"""
    rps: int = 10  # Requests per second
    max_context: int = 131072  # Maximum context length
    max_tokens_per_request: int = 2048  # Maximum tokens per request
    daily_limit: int = 10000  # Daily request limit
    quota_percentage: float = 100.0  # Percentage of total capacity

@dataclass
class TenantPolicy:
    """Tenant policy configuration"""
    priority: str = "normal"  # low, normal, high
    allow_streaming: bool = True  # Allow streaming responses
    max_concurrent_requests: int = 5  # Maximum concurrent requests
    timeout_seconds: int = 30  # Request timeout

@dataclass
class Tenant:
    """Tenant configuration"""
    alias: str
    name: str
    api_key: Optional[str] = None
    quota: Optional[TenantQuota] = None
    policy: Optional[TenantPolicy] = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    is_active: bool = True
    endpoint: Optional[str] = None

class TenantManager:
    """Runtime tenant management with CLI commands"""
    
    def __init__(self, tenants_file: str = "current_tenants.json"):
        self.tenants_file = Path(tenants_file)
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_stats: Dict[str, Dict[str, Any]] = {}
        self.base_endpoint = self.get_network_endpoint()
        
        # Load existing tenants
        self.load_tenants()
    
    def get_network_endpoint(self) -> str:
        """Get the network endpoint for the server"""
        try:
            import socket
            # Get the primary network interface IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to a remote address to determine local IP
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            
            # Use the detected IP with port 8000
            endpoint = f"http://{local_ip}:8000"
            console.print(f"[blue]🌐 Using network endpoint: {endpoint}[/blue]")
            return endpoint
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not detect network IP, using localhost: {e}[/yellow]")
            return "http://localhost:8000"
    
    def load_tenants(self):
        """Load tenants from JSON file"""
        if not self.tenants_file.exists():
            console.print(f"[yellow]⚠️ Tenants file not found: {self.tenants_file}[/yellow]")
            self.create_default_tenant()
            return
        
        try:
            with open(self.tenants_file, 'r') as f:
                data = json.load(f)
            
            tenants_data = data.get('tenants', {})
            
            for alias, tenant_data in tenants_data.items():
                # Convert dict to Tenant object
                quota_data = tenant_data.get('quota', {})
                policy_data = tenant_data.get('policy', {})
                
                quota = TenantQuota(**quota_data) if quota_data else TenantQuota()
                policy = TenantPolicy(**policy_data) if policy_data else TenantPolicy()
                
                tenant = Tenant(
                    alias=alias,
                    name=tenant_data.get('name', alias),
                    api_key=tenant_data.get('api_key'),
                    quota=quota,
                    policy=policy,
                    created_at=tenant_data.get('created_at', time.time()),
                    last_accessed=tenant_data.get('last_accessed', 0.0),
                    is_active=tenant_data.get('is_active', True),
                    endpoint=tenant_data.get('endpoint')
                )
                
                self.tenants[alias] = tenant
                self.tenant_stats[alias] = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'current_rps': 0.0,
                    'last_request_time': 0.0,
                    'error_count': 0
                }
            
            console.print(f"[green]✅ Loaded {len(self.tenants)} tenant(s)[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load tenants: {e}[/red]")
            self.create_default_tenant()
    
    def save_tenants(self):
        """Save tenants to JSON file"""
        try:
            data = {
                'tenants': {},
                'metadata': {
                    'version': '1.0.0',
                    'last_updated': time.time(),
                    'total_tenants': len(self.tenants)
                }
            }
            
            for alias, tenant in self.tenants.items():
                tenant_data = {
                    'name': tenant.name,
                    'api_key': tenant.api_key,
                    'quota': asdict(tenant.quota) if tenant.quota else {},
                    'policy': asdict(tenant.policy) if tenant.policy else {},
                    'created_at': tenant.created_at,
                    'last_accessed': tenant.last_accessed,
                    'is_active': tenant.is_active,
                    'endpoint': tenant.endpoint
                }
                data['tenants'][alias] = tenant_data
            
            with open(self.tenants_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            console.print(f"[green]✅ Saved {len(self.tenants)} tenant(s) to {self.tenants_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save tenants: {e}[/red]")
    
    def create_default_tenant(self):
        """Create default tenant at startup"""
        console.print("[blue]🔧 Creating default tenant...[/blue]")
        
        default_tenant = Tenant(
            alias="tenant_default",
            name="Default Tenant",
            api_key=None,  # No API key required for default
            quota=TenantQuota(rps=10, max_context=131072, max_tokens_per_request=2048, daily_limit=10000, quota_percentage=100.0),
            policy=TenantPolicy(priority="normal", allow_streaming=True, max_concurrent_requests=5, timeout_seconds=30),
            created_at=time.time(),
            is_active=True,
            endpoint=f"{self.base_endpoint}"
        )
        
        self.tenants["tenant_default"] = default_tenant
        self.tenant_stats["tenant_default"] = {
            'total_requests': 0,
            'total_tokens': 0,
            'current_rps': 0.0,
            'last_request_time': 0.0,
            'error_count': 0
        }
        
        # Save default configuration
        self.save_tenants()
    
    def add_tenant(self, alias: str, quota_percentage: float = 50.0, priority: str = "normal") -> bool:
        """Add a new tenant with alias ID"""
        if alias in self.tenants:
            console.print(f"[red]❌ Tenant '{alias}' already exists![/red]")
            return False
        
        # Generate API key
        api_key = self.generate_api_key()
        
        # Create tenant with specified quota
        tenant = Tenant(
            alias=alias,
            name=alias.replace('_', ' ').title(),
            api_key=api_key,
            quota=TenantQuota(
                rps=10,
                max_context=131072,
                max_tokens_per_request=2048,
                daily_limit=10000,
                quota_percentage=quota_percentage
            ),
            policy=TenantPolicy(
                priority=priority,
                allow_streaming=True,
                max_concurrent_requests=5,
                timeout_seconds=30
            ),
            created_at=time.time(),
            is_active=True,
            endpoint=f"{self.base_endpoint}/{alias}"
        )
        
        self.tenants[alias] = tenant
        self.tenant_stats[alias] = {
            'total_requests': 0,
            'total_tokens': 0,
            'current_rps': 0.0,
            'last_request_time': 0.0,
            'error_count': 0
        }
        
        console.print(f"[green]✅ Added tenant '{alias}' with {quota_percentage}% quota, {priority} priority[/green]")
        console.print(f"[blue]🔑 API Key: {api_key}[/blue]")
        console.print(f"[blue]🌐 Endpoint: {tenant.endpoint}[/blue]")
        
        return True
    
    def remove_tenant(self, alias: str) -> bool:
        """Remove tenant alias and free quota"""
        if alias not in self.tenants:
            console.print(f"[red]❌ Tenant '{alias}' not found![/red]")
            return False
        
        if alias == "tenant_default":
            console.print(f"[red]❌ Cannot remove default tenant![/red]")
            return False
        
        del self.tenants[alias]
        if alias in self.tenant_stats:
            del self.tenant_stats[alias]
        
        console.print(f"[green]✅ Removed tenant '{alias}'[/green]")
        return True
    
    def update_tenant(self, alias: str, quota_percentage: Optional[float] = None, priority: Optional[str] = None) -> bool:
        """Update existing tenant's policy live"""
        if alias not in self.tenants:
            console.print(f"[red]❌ Tenant '{alias}' not found![/red]")
            return False
        
        tenant = self.tenants[alias]
        
        if quota_percentage is not None:
            if tenant.quota:
                tenant.quota.quota_percentage = quota_percentage
            console.print(f"[blue]📊 Updated quota to {quota_percentage}%[/blue]")
        
        if priority is not None:
            if tenant.policy:
                tenant.policy.priority = priority
            console.print(f"[blue]⚡ Updated priority to {priority}[/blue]")
        
        console.print(f"[green]✅ Updated tenant '{alias}'[/green]")
        return True
    
    def list_tenants(self):
        """Display all tenants, quotas, priorities, and endpoints"""
        if not self.tenants:
            console.print("[yellow]⚠️ No tenants configured[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Active Tenants")
        table.add_column("Alias", style="cyan", width=15)
        table.add_column("Name", style="white", width=20)
        table.add_column("Quota %", style="green", width=10)
        table.add_column("Priority", style="yellow", width=10)
        table.add_column("RPS", style="blue", width=8)
        table.add_column("Endpoint", style="magenta", width=30)
        table.add_column("Status", style="white", width=10)
        
        for tenant in self.tenants.values():
            quota_pct = f"{tenant.quota.quota_percentage}%" if tenant.quota else "N/A"
            rps = tenant.quota.rps if tenant.quota else "N/A"
            priority = tenant.policy.priority if tenant.policy else "N/A"
            status = "🟢 Active" if tenant.is_active else "🔴 Inactive"
            
            table.add_row(
                tenant.alias,
                tenant.name,
                quota_pct,
                priority,
                str(rps),
                tenant.endpoint or "N/A",
                status
            )
        
        console.print(table)
    
    def get_tenant(self, alias: str) -> Optional[Tenant]:
        """Get a tenant by alias"""
        return self.tenants.get(alias)
    
    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get a tenant by API key"""
        for tenant in self.tenants.values():
            if tenant.api_key == api_key:
                return tenant
        return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def validate_tenant_request(self, alias: str, request_size: int) -> Tuple[bool, str]:
        """Validate if a tenant can make a request"""
        tenant = self.get_tenant(alias)
        if not tenant:
            return False, "Tenant not found"
        
        if not tenant.is_active:
            return False, "Tenant is inactive"
        
        if not tenant.quota:
            return True, "No quota restrictions"
        
        # Check RPS limit
        stats = self.tenant_stats.get(alias, {})
        current_time = time.time()
        last_request_time = stats.get('last_request_time', 0)
        
        if current_time - last_request_time < 1.0 / tenant.quota.rps:
            return False, "Rate limit exceeded"
        
        # Check context size
        if request_size > tenant.quota.max_context:
            return False, "Request too large"
        
        # Check daily limit
        if stats.get('total_requests', 0) >= tenant.quota.daily_limit:
            return False, "Daily limit exceeded"
        
        return True, "Valid request"
    
    def record_request(self, alias: str, tokens: int, success: bool = True):
        """Record a request for quota tracking"""
        if alias not in self.tenant_stats:
            return
        
        stats = self.tenant_stats[alias]
        current_time = time.time()
        
        stats['total_requests'] += 1
        stats['total_tokens'] += tokens
        stats['last_request_time'] = current_time
        
        if not success:
            stats['error_count'] += 1
        
        # Update RPS calculation
        if stats['last_request_time'] > 0:
            time_diff = current_time - stats['last_request_time']
            if time_diff > 0:
                stats['current_rps'] = 1.0 / time_diff
    
    def get_system_summary(self) -> Dict:
        """Get a complete system summary for the summary panel"""
        return {
            "tenant_count": len(self.tenants),
            "active_tenants": len([t for t in self.tenants.values() if t.is_active]),
            "tenants": [
                {
                    "alias": tenant.alias,
                    "name": tenant.name,
                    "quota_percentage": tenant.quota.quota_percentage if tenant.quota else 0,
                    "priority": tenant.policy.priority if tenant.policy else "normal",
                    "rps": tenant.quota.rps if tenant.quota else 0,
                    "endpoint": tenant.endpoint,
                    "is_active": tenant.is_active
                }
                for tenant in self.tenants.values()
            ]
        }
    
    def status(self):
        """Display tenant statistics"""
        if not self.tenant_stats:
            console.print("[yellow]⚠️ No tenant statistics available[/yellow]")
            return
        
        table = Table(show_header=True, box=None)
        table.add_column("Tenant", style="cyan", width=15)
        table.add_column("Total Requests", style="blue", width=15)
        table.add_column("Total Tokens", style="green", width=15)
        table.add_column("Current RPS", style="yellow", width=12)
        table.add_column("Error Count", style="red", width=12)
        table.add_column("Last Request", style="white", width=20)
        
        for alias, stats in self.tenant_stats.items():
            last_request = time.ctime(stats['last_request_time']) if stats['last_request_time'] > 0 else "Never"
            
            table.add_row(
                alias,
                f"{stats['total_requests']:,}",
                f"{stats['total_tokens']:,}",
                f"{stats['current_rps']:.2f}",
                str(stats['error_count']),
                last_request
            )
        
        console.print(table)

# CLI Commands using Typer
app = typer.Typer(help="Multi-Tenant Inference Tenant Manager")

# Global tenant manager instance
tenant_manager = None

@app.command()
def add(
    alias: str = typer.Argument(..., help="Tenant alias ID"),
    quota: float = typer.Option(50.0, "--quota", "-q", help="Quota percentage (0-100)"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority level (low/normal/high)")
):
    """Add a new tenant with alias ID"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    if priority not in ["low", "normal", "high"]:
        console.print("[red]❌ Priority must be 'low', 'normal', or 'high'[/red]")
        raise typer.Exit(1)
    
    if not 0 <= quota <= 100:
        console.print("[red]❌ Quota must be between 0 and 100[/red]")
        raise typer.Exit(1)
    
    success = tenant_manager.add_tenant(alias, quota, priority)
    if success:
        tenant_manager.save_tenants()
    else:
        raise typer.Exit(1)

@app.command()
def remove(
    alias: str = typer.Argument(..., help="Tenant alias ID to remove")
):
    """Remove tenant alias and free quota"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    success = tenant_manager.remove_tenant(alias)
    if success:
        tenant_manager.save_tenants()
    else:
        raise typer.Exit(1)

@app.command()
def update(
    alias: str = typer.Argument(..., help="Tenant alias ID to update"),
    quota: Optional[float] = typer.Option(None, "--quota", "-q", help="New quota percentage (0-100)"),
    priority: Optional[str] = typer.Option(None, "--priority", "-p", help="New priority level (low/normal/high)")
):
    """Update existing tenant's policy live"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    if priority and priority not in ["low", "normal", "high"]:
        console.print("[red]❌ Priority must be 'low', 'normal', or 'high'[/red]")
        raise typer.Exit(1)
    
    if quota is not None and not 0 <= quota <= 100:
        console.print("[red]❌ Quota must be between 0 and 100[/red]")
        raise typer.Exit(1)
    
    success = tenant_manager.update_tenant(alias, quota, priority)
    if success:
        tenant_manager.save_tenants()
    else:
        raise typer.Exit(1)

@app.command()
def list():
    """Display all tenants, quotas, priorities, and endpoints"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    tenant_manager.list_tenants()

@app.command()
def status():
    """Show tenant statistics and system status"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    console.print("[bold cyan]Tenant Statistics[/bold cyan]")
    console.print("=" * 50)
    
    if not tenant_manager.tenant_stats:
        console.print("[yellow]⚠️ No tenant statistics available[/yellow]")
        return
    
    table = Table(show_header=True, box=None)
    table.add_column("Tenant", style="cyan", width=15)
    table.add_column("Total Requests", style="blue", width=15)
    table.add_column("Total Tokens", style="green", width=15)
    table.add_column("Current RPS", style="yellow", width=12)
    table.add_column("Error Count", style="red", width=12)
    table.add_column("Last Request", style="white", width=20)
    
    for alias, stats in tenant_manager.tenant_stats.items():
        last_request = time.ctime(stats['last_request_time']) if stats['last_request_time'] > 0 else "Never"
        
        table.add_row(
            alias,
            f"{stats['total_requests']:,}",
            f"{stats['total_tokens']:,}",
            f"{stats['current_rps']:.2f}",
            str(stats['error_count']),
            last_request
        )
    
    console.print(table)

@app.command()
def infer(
    tenant: str = typer.Argument(..., help="Tenant alias ID"),
    prompt: str = typer.Argument(..., help="Text prompt to send to the model"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Temperature for generation (0.0-2.0)")
):
    """Run inference as a tenant from CLI (Shellfish / SSH)"""
    global tenant_manager
    if not tenant_manager:
        tenant_manager = TenantManager()
    
    # Get tenant info
    tenant_obj = tenant_manager.get_tenant(tenant)
    if not tenant_obj:
        console.print(f"[red]❌ Tenant '{tenant}' not found[/red]")
        raise typer.Exit(1)
    
    if not tenant_obj.is_active:
        console.print(f"[red]❌ Tenant '{tenant}' is inactive[/red]")
        raise typer.Exit(1)
    
    # Prepare request - all tenants use the same router endpoint
    base_url = "http://localhost:8000"
    api_key = tenant_obj.api_key
    
    headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenant
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Validate request size
    is_valid, message = tenant_manager.validate_tenant_request(tenant, len(prompt))
    if not is_valid:
        console.print(f"[red]❌ Request validation failed: {message}[/red]")
        raise typer.Exit(1)
    
    try:
        console.print(f"[blue]🚀 Sending request to {base_url}/v1/chat/completions...[/blue]")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-oss-20b-F16",  # Use the detected model
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the response text from chat completions format
        response_text = ""
        if 'choices' in data and len(data['choices']) > 0:
            response_text = data['choices'][0].get('message', {}).get('content', '')
        
        # Record successful request
        tokens_generated = len(response_text.split()) if response_text else max_tokens
        tenant_manager.record_request(tenant, tokens_generated, success=True)
        
        console.print("[green]✅ Response:[/green]")
        if response_text:
            console.print(f"[white]{response_text}[/white]")
        else:
            console.print(json.dumps(data, indent=2))
        
    except requests.exceptions.RequestException as e:
        # Record failed request
        tenant_manager.record_request(tenant, 0, success=False)
        console.print(f"[red]❌ Request failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        # Record failed request
        tenant_manager.record_request(tenant, 0, success=False)
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point for CLI"""
    app()

if __name__ == "__main__":
    main()
