# inference/tui.py

import asyncio
import aiohttp
import time
import json
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich import print
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.align import Align
from rich.columns import Columns
import signal
import sys

console = Console()

class InferenceDashboard:
    """Rich-based TUI dashboard for multi-tenant inference monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.running = False
        self.metrics = {
            'requests_per_second': 0.0,
            'latency_p50': 0.0,
            'latency_p95': 0.0,
            'tokens_per_second': 0.0,
            'active_tenants': 0,
            'queue_depth': 0,
            'gpu_utilization': {},
            'instance_status': {},
            'error_rate': 0.0
        }
        self.update_interval = 2.0  # seconds
        
    async def start(self):
        """Start the dashboard"""
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        console.print("[bold green]📊 Starting Multi-Tenant Inference Dashboard[/bold green]")
        console.print("[dim]Press Ctrl+C to exit[/dim]\n")
        
        # Create layout
        layout = self.create_layout()
        
        try:
            with Live(layout, refresh_per_second=0.5, screen=True) as live:
                while self.running:
                    try:
                        # Update metrics
                        await self.update_metrics()
                        
                        # Update layout
                        layout = self.create_layout()
                        live.update(layout)
                        
                        await asyncio.sleep(self.update_interval)
                        
                    except Exception as e:
                        console.print(f"[red]❌ Dashboard error: {e}[/red]")
                        await asyncio.sleep(1.0)
                        
        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]⚠️ Dashboard stopped by user[/yellow]")
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout"""
        layout = Layout()
        
        # Split into main areas
        layout.split_column(
            Layout(self.create_header(), size=3),
            Layout(self.create_main_content()),
            Layout(self.create_footer(), size=3)
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """Create the header panel"""
        title = Text("Multi-Tenant Inference Dashboard", style="bold cyan")
        subtitle = Text(f"Monitoring {self.base_url}", style="dim")
        
        header_text = Text()
        header_text.append(title)
        header_text.append("\n")
        header_text.append(subtitle)
        
        return Panel(
            Align.center(header_text),
            border_style="cyan",
            padding=(0, 1)
        )
    
    def create_main_content(self) -> Layout:
        """Create the main content area"""
        layout = Layout()
        layout.split_row(
            Layout(self.create_metrics_panel(), ratio=1),
            Layout(self.create_instances_panel(), ratio=1),
            Layout(self.create_gpu_panel(), ratio=1)
        )
        
        return layout
    
    def create_metrics_panel(self) -> Panel:
        """Create the metrics panel"""
        table = Table(show_header=True, box=None, title="📊 Performance Metrics")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=15)
        table.add_column("Unit", style="green", width=8)
        
        table.add_row("Requests/sec", f"{self.metrics['requests_per_second']:.2f}", "req/s")
        table.add_row("Latency P50", f"{self.metrics['latency_p50']:.3f}", "s")
        table.add_row("Latency P95", f"{self.metrics['latency_p95']:.3f}", "s")
        table.add_row("Tokens/sec", f"{self.metrics['tokens_per_second']:.2f}", "tokens/s")
        table.add_row("Active Tenants", str(self.metrics['active_tenants']), "tenants")
        table.add_row("Queue Depth", str(self.metrics['queue_depth']), "requests")
        table.add_row("Error Rate", f"{self.metrics['error_rate']:.2f}", "%")
        
        return Panel(table, border_style="blue")
    
    def create_instances_panel(self) -> Panel:
        """Create the instances status panel"""
        table = Table(show_header=True, box=None, title="🖥️ Instance Status")
        table.add_column("Instance", style="cyan", width=8)
        table.add_column("Port", style="white", width=6)
        table.add_column("Status", style="white", width=10)
        table.add_column("Requests", style="blue", width=10)
        table.add_column("Latency", style="yellow", width=10)
        
        for i, status in self.metrics['instance_status'].items():
            status_icon = "🟢" if status.get('healthy', False) else "🔴"
            table.add_row(
                f"GPU {i}",
                str(status.get('port', 'N/A')),
                f"{status_icon} {status.get('status', 'Unknown')}",
                f"{status.get('total_requests', 0):,}",
                f"{status.get('avg_latency', 0):.3f}s"
            )
        
        return Panel(table, border_style="green")
    
    def create_gpu_panel(self) -> Panel:
        """Create the GPU utilization panel"""
        table = Table(show_header=True, box=None, title="🎮 GPU Utilization")
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Utilization", style="white", width=12)
        table.add_column("Memory", style="blue", width=12)
        table.add_column("Temperature", style="red", width=12)
        
        for gpu_id, gpu_info in self.metrics['gpu_utilization'].items():
            utilization = gpu_info.get('utilization', 0)
            memory_used = gpu_info.get('memory_used', 0)
            memory_total = gpu_info.get('memory_total', 1)
            temperature = gpu_info.get('temperature', 0)
            
            # Create progress bar for utilization
            util_progress = Progress(
                BarColumn(bar_width=10),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            )
            util_task = util_progress.add_task("", total=100, completed=utilization)
            
            table.add_row(
                f"GPU {gpu_id}",
                f"{utilization}%",
                f"{memory_used:.1f}GB/{memory_total:.1f}GB",
                f"{temperature}°C"
            )
        
        return Panel(table, border_style="yellow")
    
    def create_footer(self) -> Panel:
        """Create the footer panel"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "🟢 Running" if self.running else "🔴 Stopped"
        
        footer_text = Text()
        footer_text.append(f"Status: {status}", style="bold")
        footer_text.append(" | ")
        footer_text.append(f"Last Update: {current_time}", style="dim")
        footer_text.append(" | ")
        footer_text.append("Press Ctrl+C to exit", style="dim")
        
        return Panel(
            Align.center(footer_text),
            border_style="dim",
            padding=(0, 1)
        )
    
    async def update_metrics(self):
        """Update metrics from the inference API"""
        try:
            # Get health status
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self.metrics['active_tenants'] = health_data.get('healthy_instances', 0)
                    else:
                        self.metrics['active_tenants'] = 0
                
                # Get detailed metrics
                async with session.get(f"{self.base_url}/metrics", timeout=5) as response:
                    if response.status == 200:
                        metrics_data = await response.json()
                        self.update_metrics_from_api(metrics_data)
                    else:
                        # Use default values if metrics endpoint fails
                        self.reset_metrics()
                
        except Exception as e:
            # If API is not available, reset metrics
            self.reset_metrics()
    
    def update_metrics_from_api(self, data: Dict[str, Any]):
        """Update metrics from API response"""
        instances = data.get('instances', {})
        
        # Calculate aggregate metrics
        total_requests = sum(inst.get('total_requests', 0) for inst in instances.values())
        total_tokens = sum(inst.get('total_tokens', 0) for inst in instances.values())
        avg_latencies = [inst.get('avg_latency', 0) for inst in instances.values() if inst.get('avg_latency', 0) > 0]
        
        # Update metrics
        self.metrics['requests_per_second'] = total_requests / 60.0  # Rough RPS calculation
        self.metrics['tokens_per_second'] = total_tokens / 60.0  # Rough tokens/sec calculation
        self.metrics['latency_p50'] = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0
        self.metrics['latency_p95'] = self.metrics['latency_p50'] * 1.5  # Rough P95 estimate
        self.metrics['queue_depth'] = sum(inst.get('active_requests', 0) for inst in instances.values())
        
        # Calculate error rate
        total_errors = sum(inst.get('error_count', 0) for inst in instances.values())
        self.metrics['error_rate'] = (total_errors / max(total_requests, 1)) * 100
        
        # Update instance status
        self.metrics['instance_status'] = {}
        for i, inst in instances.items():
            self.metrics['instance_status'][i] = {
                'port': 8080 + int(i),
                'healthy': inst.get('error_count', 0) / max(inst.get('total_requests', 1), 1) < 0.5,
                'status': 'Healthy' if inst.get('error_count', 0) / max(inst.get('total_requests', 1), 1) < 0.5 else 'Unhealthy',
                'total_requests': inst.get('total_requests', 0),
                'avg_latency': inst.get('avg_latency', 0)
            }
        
        # Update GPU metrics
        self.update_gpu_metrics()
    
    def reset_metrics(self):
        """Reset metrics to default values"""
        self.metrics.update({
            'requests_per_second': 0.0,
            'latency_p50': 0.0,
            'latency_p95': 0.0,
            'tokens_per_second': 0.0,
            'active_tenants': 0,
            'queue_depth': 0,
            'error_rate': 0.0,
            'instance_status': {},
            'gpu_utilization': {}
        })
    
    def update_gpu_metrics(self):
        """Update GPU utilization metrics using nvidia-smi"""
        try:
            # Run nvidia-smi to get GPU info
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpu_id = parts[0]
                        utilization = int(parts[1])
                        memory_used = float(parts[2]) / 1024  # Convert MB to GB
                        memory_total = float(parts[3]) / 1024  # Convert MB to GB
                        temperature = int(parts[4])
                        
                        self.metrics['gpu_utilization'][gpu_id] = {
                            'utilization': utilization,
                            'memory_used': memory_used,
                            'memory_total': memory_total,
                            'temperature': temperature
                        }
            else:
                # nvidia-smi failed, use default values
                self.metrics['gpu_utilization'] = {}
                
        except Exception:
            # nvidia-smi not available or failed, use default values
            self.metrics['gpu_utilization'] = {}
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.running = False
        console.print(f"\n[yellow]⚠️ Received signal {signum}, shutting down...[/yellow]")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Tenant Inference Dashboard")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the inference API")
    parser.add_argument("--update-interval", type=float, default=2.0, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = InferenceDashboard(base_url=args.base_url)
    dashboard.update_interval = args.update_interval
    
    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Dashboard error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
