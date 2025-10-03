# inference/summary_panel.py

import time
import json
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box

console = Console()

class SummaryPanel:
    """Live summary panel showing GPUs, models, tenants, and status"""
    
    def __init__(self, gpu_detector=None, model_detector=None, tenant_manager=None):
        self.gpu_detector = gpu_detector
        self.model_detector = model_detector
        self.tenant_manager = tenant_manager
        self.last_update = 0
        self.update_interval = 5  # seconds
        
    def check_service_status(self) -> Dict[str, bool]:
        """Check if services are running"""
        status = {
            "router": False,
            "llama_servers": 0,
            "total_instances": 0
        }
        
        # Check router (port 8000)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            status["router"] = result == 0
        except:
            pass
        
        # Check llama-server processes
        try:
            result = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
            if result.stdout.strip():
                status["llama_servers"] = len(result.stdout.strip().split('\n'))
        except:
            pass
        
        # Count total instances from profile
        if self.gpu_detector:
            status["total_instances"] = len(self.gpu_detector.get_available_gpus())
        
        return status
    
    def get_gpu_summary(self) -> Dict:
        """Get GPU summary information"""
        if not self.gpu_detector:
            return {"gpu_count": 0, "available_gpus": 0, "gpus": []}
        
        return self.gpu_detector.get_system_summary()
    
    def get_model_summary(self) -> Dict:
        """Get model summary information"""
        if not self.model_detector:
            return {"model_count": 0, "default_model": None, "models": []}
        
        return self.model_detector.get_system_summary()
    
    def get_tenant_summary(self) -> Dict:
        """Get tenant summary information"""
        if not self.tenant_manager:
            return {"tenant_count": 0, "active_tenants": 0, "tenants": []}
        
        return self.tenant_manager.get_system_summary()
    
    def create_gpu_table(self, gpu_summary: Dict) -> Table:
        """Create GPU status table"""
        table = Table(show_header=True, box=box.ROUNDED, title="GPU Status")
        table.add_column("Index", style="cyan", width=6)
        table.add_column("Name", style="white", width=25)
        table.add_column("Memory", style="green", width=12)
        table.add_column("Port", style="blue", width=8)
        table.add_column("Status", style="white", width=10)
        
        if gpu_summary["gpu_count"] == 0:
            table.add_row("N/A", "No GPUs detected", "N/A", "N/A", "❌")
        else:
            for gpu in gpu_summary["gpus"]:
                status = "🟢" if gpu["is_available"] else "🔴"
                memory = f"{gpu['memory_free_gb']:.1f}/{gpu['memory_gb']:.1f}GB"
                port = str(gpu["port"]) if gpu["port"] else "N/A"
                
                table.add_row(
                    str(gpu["index"]),
                    gpu["name"][:25],
                    memory,
                    port,
                    status
                )
        
        return table
    
    def create_model_table(self, model_summary: Dict) -> Table:
        """Create model status table"""
        table = Table(show_header=True, box=box.ROUNDED, title="Model Status")
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Size", style="green", width=10)
        table.add_column("Context", style="blue", width=10)
        table.add_column("Quant", style="yellow", width=8)
        table.add_column("Status", style="white", width=8)
        
        if model_summary["model_count"] == 0:
            table.add_row("No models", "N/A", "N/A", "N/A", "❌")
        else:
            default_model = model_summary["default_model"]
            if default_model:
                status = "⭐ Default" if default_model else "📁"
                table.add_row(
                    default_model["name"][:20],
                    f"{default_model['size_gb']:.1f}GB",
                    f"{default_model['context_window']:,}",
                    default_model["quantization"][:8],
                    status
                )
        
        return table
    
    def create_tenant_table(self, tenant_summary: Dict) -> Table:
        """Create tenant status table"""
        table = Table(show_header=True, box=box.ROUNDED, title="Tenant Status")
        table.add_column("Alias", style="cyan", width=15)
        table.add_column("Quota %", style="green", width=10)
        table.add_column("Priority", style="yellow", width=10)
        table.add_column("RPS", style="blue", width=8)
        table.add_column("Status", style="white", width=10)
        
        if tenant_summary["tenant_count"] == 0:
            table.add_row("No tenants", "N/A", "N/A", "N/A", "❌")
        else:
            for tenant in tenant_summary["tenants"]:
                status = "🟢" if tenant["is_active"] else "🔴"
                table.add_row(
                    tenant["alias"],
                    f"{tenant['quota_percentage']}%",
                    tenant["priority"],
                    str(tenant["rps"]),
                    status
                )
        
        return table
    
    def create_service_table(self, service_status: Dict) -> Table:
        """Create service status table"""
        table = Table(show_header=True, box=box.ROUNDED, title="Service Status")
        table.add_column("Service", style="cyan", width=15)
        table.add_column("Status", style="white", width=10)
        table.add_column("Details", style="blue", width=20)
        
        # Router status
        router_status = "🟢 Running" if service_status["router"] else "🔴 Stopped"
        table.add_row("Router", router_status, "Port 8000")
        
        # Llama servers
        llama_count = service_status["llama_servers"]
        total_instances = service_status["total_instances"]
        llama_status = f"🟢 {llama_count}/{total_instances}" if llama_count > 0 else "🔴 0/0"
        table.add_row("Llama Servers", llama_status, f"Instances running")
        
        return table
    
    def create_summary_layout(self) -> Layout:
        """Create the main summary layout"""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main into columns
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Split left column
        layout["left"].split_column(
            Layout(name="gpu", ratio=1),
            Layout(name="model", ratio=1)
        )
        
        # Split right column
        layout["right"].split_column(
            Layout(name="tenant", ratio=1),
            Layout(name="service", ratio=1)
        )
        
        return layout
    
    def update_summary(self) -> Layout:
        """Update the summary panel with current data"""
        layout = self.create_summary_layout()
        
        # Get current data
        gpu_summary = self.get_gpu_summary()
        model_summary = self.get_model_summary()
        tenant_summary = self.get_tenant_summary()
        service_status = self.check_service_status()
        
        # Create tables
        gpu_table = self.create_gpu_table(gpu_summary)
        model_table = self.create_model_table(model_summary)
        tenant_table = self.create_tenant_table(tenant_summary)
        service_table = self.create_service_table(service_status)
        
        # Update layout
        layout["header"] = Panel(
            Align.center(Text("Multi-Tenant Inference Control Panel", style="bold cyan")),
            box=box.DOUBLE
        )
        
        layout["gpu"] = gpu_table
        layout["model"] = model_table
        layout["tenant"] = tenant_table
        layout["service"] = service_table
        
        # Footer with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        layout["footer"] = Panel(
            Align.center(Text(f"Last updated: {timestamp}", style="dim")),
            box=box.ROUNDED
        )
        
        return layout
    
    def display_static_summary(self):
        """Display a static summary (non-live)"""
        console.print("\n" + "="*80)
        console.print("[bold cyan]Multi-Tenant Inference Control Panel[/bold cyan]")
        console.print("="*80)
        
        # Get current data
        gpu_summary = self.get_gpu_summary()
        model_summary = self.get_model_summary()
        tenant_summary = self.get_tenant_summary()
        service_status = self.check_service_status()
        
        # Display tables
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"• GPUs detected: {gpu_summary['gpu_count']} (mapped to ports {list(gpu_summary['port_mapping'].values()) if gpu_summary['port_mapping'] else 'N/A'})")
        
        if model_summary['default_model']:
            context_window = model_summary['default_model'].get('context_window', 0)
            if isinstance(context_window, (int, float)):
                console.print(f"• Model: {model_summary['default_model']['name']} (ctx: {context_window:,})")
            else:
                console.print(f"• Model: {model_summary['default_model']['name']} (ctx: {context_window})")
        else:
            console.print("• Model: No model detected")
        
        # Enhanced tenant summary
        console.print(f"• Tenants: {tenant_summary['tenant_count']} ({tenant_summary['active_tenants']} active)")
        for tenant in tenant_summary['tenants']:
            status_icon = "🟢" if tenant['is_active'] else "🔴"
            console.print(f"  - {tenant['alias']} (quota={tenant['quota_percentage']}%, priority={tenant['priority']}) → {tenant['endpoint']} {status_icon}")
        
        # Status with guidance hints
        if service_status["router"]:
            router_status = "🟢 Running"
        else:
            router_status = "🔴 Stopped (press 1 to Start Services)"
        
        if service_status["llama_servers"] > 0:
            llama_status = f"🟢 {service_status['llama_servers']}/{service_status['total_instances']} running"
        else:
            llama_status = "🔴 0/0 (waiting for startup)"
        
        console.print(f"• Status: Router {router_status}, Llama Servers {llama_status}")
        
        # Display detailed tables
        console.print("\n[bold]Quick Actions:[/bold]")
        console.print("1. Start/Stop Services")
        console.print("2. Monitor (live dashboard)")
        console.print("3. Tenant Manager (add/remove/list aliases)")
        console.print("4. Advanced Options")
        console.print("0. Exit Wizard")
        
        console.print("\n" + "="*80)
    
    def run_live_dashboard(self, refresh_interval: float = 2.0):
        """Run live dashboard with auto-refresh"""
        try:
            with Live(self.update_summary(), refresh_per_second=1/refresh_interval, screen=True) as live:
                while True:
                    live.update(self.update_summary())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Live dashboard stopped[/yellow]")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status with real GPU metrics"""
        gpu_summary = self.get_gpu_summary()
        model_summary = self.get_model_summary()
        tenant_summary = self.get_tenant_summary()
        service_status = self.check_service_status()
        
        # Get real GPU metrics if available
        gpu_health_info = None
        if self.gpu_detector:
            try:
                from inference.monitor import MonitoringDashboard
                monitor = MonitoringDashboard(self.gpu_detector, self.tenant_manager)
                gpu_metrics = monitor.get_gpu_metrics()
                
                if gpu_metrics:
                    # Calculate average GPU utilization and memory usage
                    avg_utilization = sum(gpu.utilization_percent for gpu in gpu_metrics) / len(gpu_metrics)
                    avg_memory_used = sum(gpu.memory_used_gb for gpu in gpu_metrics) / len(gpu_metrics)
                    avg_memory_total = sum(gpu.memory_total_gb for gpu in gpu_metrics) / len(gpu_metrics)
                    avg_temperature = sum(gpu.temperature_c for gpu in gpu_metrics if gpu.temperature_c > 0) / max(1, len([g for g in gpu_metrics if g.temperature_c > 0]))
                    
                    gpu_health_info = {
                        "utilization_percent": avg_utilization,
                        "memory_used_gb": avg_memory_used,
                        "memory_total_gb": avg_memory_total,
                        "temperature_c": avg_temperature if avg_temperature > 0 else None
                    }
            except Exception:
                # Fallback to basic health calculation
                pass
        
        # Calculate health score
        health_score = 0
        max_score = 4
        
        # GPU health
        if gpu_summary["gpu_count"] > 0 and gpu_summary["available_gpus"] > 0:
            health_score += 1
        
        # Model health
        if model_summary["model_count"] > 0 and model_summary["default_model"]:
            health_score += 1
        
        # Tenant health
        if tenant_summary["tenant_count"] > 0 and tenant_summary["active_tenants"] > 0:
            health_score += 1
        
        # Service health
        if service_status["router"] and service_status["llama_servers"] > 0:
            health_score += 1
        
        health_percentage = (health_score / max_score) * 100
        
        return {
            "health_score": health_score,
            "max_score": max_score,
            "health_percentage": health_percentage,
            "status": "healthy" if health_percentage >= 75 else "degraded" if health_percentage >= 50 else "unhealthy",
            "gpu_health_info": gpu_health_info,
            "gpu_summary": gpu_summary,
            "model_summary": model_summary,
            "tenant_summary": tenant_summary,
            "service_status": service_status
        }

def main():
    """Test summary panel"""
    from gpu_detector import GPUDetector
    from model_detector import ModelDetector
    from tenant_manager import TenantManager
    
    # Initialize components
    gpu_detector = GPUDetector()
    model_detector = ModelDetector()
    tenant_manager = TenantManager()
    
    # Detect systems
    gpu_detector.detect_gpus()
    gpu_detector.create_port_mapping()
    model_detector.scan_models()
    
    # Create summary panel
    panel = SummaryPanel(gpu_detector, model_detector, tenant_manager)
    
    # Display static summary
    panel.display_static_summary()
    
    # Show system health
    health = panel.get_system_health()
    console.print(f"\n[bold]System Health: {health['health_percentage']:.1f}% ({health['status']})[/bold]")

if __name__ == "__main__":
    main()
