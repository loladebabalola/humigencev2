# inference/monitor.py

import os
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

@dataclass
class SystemMetrics:
    """System metrics container"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    timestamp: float

@dataclass
class GPUMetrics:
    """GPU metrics container"""
    index: int
    name: str
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    utilization_percent: float
    temperature_c: float
    power_usage_w: float
    timestamp: float

@dataclass
class ServiceMetrics:
    """Service metrics container"""
    service_name: str
    status: str
    pid: Optional[int]
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    requests_total: int
    requests_per_second: float
    avg_latency_ms: float
    error_rate: float
    timestamp: float

class MonitoringDashboard:
    """Live monitoring dashboard with comprehensive metrics"""
    
    def __init__(self, gpu_detector=None, tenant_manager=None):
        self.gpu_detector = gpu_detector
        self.tenant_manager = tenant_manager
        self.metrics_history = {
            "system": [],
            "gpu": [],
            "services": [],
            "tenants": []
        }
        self.max_history = 100  # Keep last 100 data points
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                timestamp=time.time()
            )
        except Exception as e:
            console.print(f"[red]❌ Error getting system metrics: {e}[/red]")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, time.time())
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics"""
        gpu_metrics = []
        
        try:
            # Try to get GPU info using nvidia-ml-py or nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            try:
                                index = int(parts[0])
                                name = parts[1]
                                memory_used = float(parts[2]) / 1024  # Convert MB to GB
                                memory_total = float(parts[3]) / 1024
                                utilization = float(parts[4])
                                temperature = float(parts[5])
                                power_usage = float(parts[6])
                                
                                gpu_metrics.append(GPUMetrics(
                                    index=index,
                                    name=name,
                                    memory_used_gb=memory_used,
                                    memory_total_gb=memory_total,
                                    memory_percent=(memory_used / memory_total) * 100,
                                    utilization_percent=utilization,
                                    temperature_c=temperature,
                                    power_usage_w=power_usage,
                                    timestamp=time.time()
                                ))
                            except (ValueError, IndexError):
                                continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to basic GPU detection
            if self.gpu_detector:
                for gpu in self.gpu_detector.get_available_gpus():
                    gpu_metrics.append(GPUMetrics(
                        index=gpu.index,
                        name=gpu.name,
                        memory_used_gb=gpu.memory_used_gb,
                        memory_total_gb=gpu.memory_gb,
                        memory_percent=gpu.utilization_percent,
                        utilization_percent=gpu.utilization_percent,
                        temperature_c=0.0,  # Not available without nvidia-smi
                        power_usage_w=0.0,  # Not available without nvidia-smi
                        timestamp=time.time()
                    ))
        
        return gpu_metrics
    
    def get_service_metrics(self) -> List[ServiceMetrics]:
        """Get current service metrics"""
        services = []
        
        # Check for router process
        router_pid = self._get_process_pid("router.py")
        if router_pid:
            services.append(self._get_process_metrics("Router", router_pid))
        
        # Check for supervisor process
        supervisor_pid = self._get_process_pid("supervisor.py")
        if supervisor_pid:
            services.append(self._get_process_metrics("Supervisor", supervisor_pid))
        
        # Check for llama-server processes
        llama_pids = self._get_process_pids("llama-server")
        for i, pid in enumerate(llama_pids):
            services.append(self._get_process_metrics(f"Llama-Server-{i+1}", pid))
        
        return services
    
    def _get_process_pid(self, process_name: str) -> Optional[int]:
        """Get PID of a process by name"""
        try:
            result = subprocess.run(['pgrep', '-f', process_name], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return None
    
    def _get_process_pids(self, process_name: str) -> List[int]:
        """Get PIDs of all processes by name"""
        pids = []
        try:
            result = subprocess.run(['pgrep', '-f', process_name], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        pids.append(int(line.strip()))
        except:
            pass
        return pids
    
    def _get_process_metrics(self, service_name: str, pid: int) -> ServiceMetrics:
        """Get metrics for a specific process"""
        try:
            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)  # Convert to MB
            uptime = time.time() - process.create_time()
            
            return ServiceMetrics(
                service_name=service_name,
                status="running",
                pid=pid,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                uptime_seconds=uptime,
                requests_total=0,  # Would need to track this separately
                requests_per_second=0.0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                timestamp=time.time()
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return ServiceMetrics(
                service_name=service_name,
                status="stopped",
                pid=None,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                requests_total=0,
                requests_per_second=0.0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                timestamp=time.time()
            )
    
    def get_tenant_metrics(self) -> List[Dict[str, Any]]:
        """Get current tenant metrics"""
        if not self.tenant_manager:
            return []
        
        tenant_metrics = []
        for alias, stats in self.tenant_manager.tenant_stats.items():
            tenant_metrics.append({
                "alias": alias,
                "total_requests": stats.get("total_requests", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "current_rps": stats.get("current_rps", 0.0),
                "error_count": stats.get("error_count", 0),
                "last_request_time": stats.get("last_request_time", 0)
            })
        
        return tenant_metrics
    
    def create_system_table(self, system_metrics: SystemMetrics) -> Table:
        """Create system metrics table"""
        table = Table(show_header=True, box=box.ROUNDED, title="System Metrics")
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="white", width=15)
        table.add_column("Progress", style="green", width=20)
        
        # CPU
        cpu_bar = Progress(BarColumn(bar_width=20), TextColumn("{task.percentage:>3.0f}%"))
        cpu_task = cpu_bar.add_task("CPU", total=100, completed=system_metrics.cpu_percent)
        table.add_row("CPU Usage", f"{system_metrics.cpu_percent:.1f}%", cpu_bar)
        
        # Memory
        memory_bar = Progress(BarColumn(bar_width=20), TextColumn("{task.percentage:>3.0f}%"))
        memory_task = memory_bar.add_task("Memory", total=100, completed=system_metrics.memory_percent)
        table.add_row("Memory", f"{system_metrics.memory_used_gb:.1f}/{system_metrics.memory_total_gb:.1f}GB", memory_bar)
        
        # Disk
        disk_bar = Progress(BarColumn(bar_width=20), TextColumn("{task.percentage:>3.0f}%"))
        disk_task = disk_bar.add_task("Disk", total=100, completed=system_metrics.disk_percent)
        table.add_row("Disk", f"{system_metrics.disk_used_gb:.1f}/{system_metrics.disk_total_gb:.1f}GB", disk_bar)
        
        return table
    
    def create_gpu_table(self, gpu_metrics: List[GPUMetrics]) -> Table:
        """Create GPU metrics table"""
        table = Table(show_header=True, box=box.ROUNDED, title="GPU Metrics")
        table.add_column("Index", style="cyan", width=6)
        table.add_column("Name", style="white", width=20)
        table.add_column("Memory", style="green", width=15)
        table.add_column("Utilization", style="yellow", width=15)
        table.add_column("Temperature", style="red", width=12)
        table.add_column("Power", style="blue", width=10)
        
        for gpu in gpu_metrics:
            memory_str = f"{gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f}GB"
            utilization_str = f"{gpu.utilization_percent:.1f}%"
            temp_str = f"{gpu.temperature_c:.1f}°C" if gpu.temperature_c > 0 else "N/A"
            power_str = f"{gpu.power_usage_w:.1f}W" if gpu.power_usage_w > 0 else "N/A"
            
            table.add_row(
                str(gpu.index),
                gpu.name[:20],
                memory_str,
                utilization_str,
                temp_str,
                power_str
            )
        
        return table
    
    def create_service_table(self, service_metrics: List[ServiceMetrics]) -> Table:
        """Create service metrics table"""
        table = Table(show_header=True, box=box.ROUNDED, title="Service Metrics")
        table.add_column("Service", style="cyan", width=15)
        table.add_column("Status", style="white", width=10)
        table.add_column("PID", style="blue", width=8)
        table.add_column("CPU %", style="yellow", width=8)
        table.add_column("Memory", style="green", width=10)
        table.add_column("Uptime", style="magenta", width=12)
        
        for service in service_metrics:
            status_color = "🟢" if service.status == "running" else "🔴"
            uptime_str = f"{service.uptime_seconds/3600:.1f}h" if service.uptime_seconds > 3600 else f"{service.uptime_seconds/60:.1f}m"
            
            table.add_row(
                service.service_name,
                f"{status_color} {service.status}",
                str(service.pid) if service.pid else "N/A",
                f"{service.cpu_percent:.1f}%",
                f"{service.memory_mb:.0f}MB",
                uptime_str
            )
        
        return table
    
    def create_tenant_table(self, tenant_metrics: List[Dict[str, Any]]) -> Table:
        """Create tenant metrics table"""
        table = Table(show_header=True, box=box.ROUNDED, title="Tenant Metrics")
        table.add_column("Tenant", style="cyan", width=15)
        table.add_column("Requests", style="blue", width=12)
        table.add_column("Tokens", style="green", width=12)
        table.add_column("RPS", style="yellow", width=8)
        table.add_column("Errors", style="red", width=8)
        table.add_column("Last Request", style="white", width=15)
        
        for tenant in tenant_metrics:
            last_request = time.ctime(tenant["last_request_time"]) if tenant["last_request_time"] > 0 else "Never"
            
            table.add_row(
                tenant["alias"],
                f"{tenant['total_requests']:,}",
                f"{tenant['total_tokens']:,}",
                f"{tenant['current_rps']:.2f}",
                str(tenant["error_count"]),
                last_request
            )
        
        return table
    
    def create_monitoring_layout(self) -> Layout:
        """Create the main monitoring layout"""
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
            Layout(name="system", ratio=1),
            Layout(name="gpu", ratio=1)
        )
        
        # Split right column
        layout["right"].split_column(
            Layout(name="service", ratio=1),
            Layout(name="tenant", ratio=1)
        )
        
        return layout
    
    def update_monitoring(self) -> Layout:
        """Update the monitoring dashboard with current data"""
        layout = self.create_monitoring_layout()
        
        # Get current metrics
        system_metrics = self.get_system_metrics()
        gpu_metrics = self.get_gpu_metrics()
        service_metrics = self.get_service_metrics()
        tenant_metrics = self.get_tenant_metrics()
        
        # Create tables
        system_table = self.create_system_table(system_metrics)
        gpu_table = self.create_gpu_table(gpu_metrics)
        service_table = self.create_service_table(service_metrics)
        tenant_table = self.create_tenant_table(tenant_metrics)
        
        # Update layout
        layout["header"] = Panel(
            Align.center(Text("Multi-Tenant Inference Monitoring Dashboard", style="bold cyan")),
            box=box.DOUBLE
        )
        
        layout["system"] = system_table
        layout["gpu"] = gpu_table
        layout["service"] = service_table
        layout["tenant"] = tenant_table
        
        # Footer with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        layout["footer"] = Panel(
            Align.center(Text(f"Last updated: {timestamp}", style="dim")),
            box=box.ROUNDED
        )
        
        return layout
    
    def run_live_monitoring(self, refresh_interval: float = 2.0):
        """Run live monitoring dashboard with auto-refresh"""
        try:
            with Live(self.update_monitoring(), refresh_per_second=1/refresh_interval, screen=True) as live:
                while True:
                    live.update(self.update_monitoring())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Live monitoring stopped[/yellow]")
    
    def display_static_monitoring(self):
        """Display a static monitoring view (non-live)"""
        console.print("\n" + "="*80)
        console.print("[bold cyan]Multi-Tenant Inference Monitoring Dashboard[/bold cyan]")
        console.print("="*80)
        
        # Get current metrics
        system_metrics = self.get_system_metrics()
        gpu_metrics = self.get_gpu_metrics()
        service_metrics = self.get_service_metrics()
        tenant_metrics = self.get_tenant_metrics()
        
        # Display tables
        console.print("\n[bold]System Status:[/bold]")
        console.print(f"• CPU: {system_metrics.cpu_percent:.1f}%")
        console.print(f"• Memory: {system_metrics.memory_used_gb:.1f}/{system_metrics.memory_total_gb:.1f}GB ({system_metrics.memory_percent:.1f}%)")
        console.print(f"• Disk: {system_metrics.disk_used_gb:.1f}/{system_metrics.disk_total_gb:.1f}GB ({system_metrics.disk_percent:.1f}%)")
        
        console.print(f"\n[bold]GPU Status:[/bold]")
        if gpu_metrics:
            for gpu in gpu_metrics:
                console.print(f"• GPU {gpu.index}: {gpu.name} - {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f}GB ({gpu.utilization_percent:.1f}%)")
        else:
            console.print("• No GPU metrics available")
        
        console.print(f"\n[bold]Services:[/bold]")
        for service in service_metrics:
            status = "🟢 Running" if service.status == "running" else "🔴 Stopped"
            console.print(f"• {service.service_name}: {status} (PID: {service.pid})")
        
        console.print(f"\n[bold]Tenants:[/bold]")
        for tenant in tenant_metrics:
            console.print(f"• {tenant['alias']}: {tenant['total_requests']} requests, {tenant['current_rps']:.2f} RPS")
        
        console.print("\n" + "="*80)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        system_metrics = self.get_system_metrics()
        gpu_metrics = self.get_gpu_metrics()
        service_metrics = self.get_service_metrics()
        
        # Calculate health score
        health_score = 0
        max_score = 3
        
        # System health
        if system_metrics.cpu_percent < 80 and system_metrics.memory_percent < 90:
            health_score += 1
        
        # GPU health
        if gpu_metrics and all(gpu.utilization_percent < 95 for gpu in gpu_metrics):
            health_score += 1
        
        # Service health
        running_services = sum(1 for s in service_metrics if s.status == "running")
        if running_services > 0:
            health_score += 1
        
        health_percentage = (health_score / max_score) * 100
        
        return {
            "health_score": health_score,
            "max_score": max_score,
            "health_percentage": health_percentage,
            "status": "healthy" if health_percentage >= 75 else "degraded" if health_percentage >= 50 else "unhealthy",
            "system_metrics": system_metrics,
            "gpu_metrics": gpu_metrics,
            "service_metrics": service_metrics
        }

def main():
    """Test monitoring dashboard"""
    from gpu_detector import GPUDetector
    from tenant_manager import TenantManager
    
    # Initialize components
    gpu_detector = GPUDetector()
    tenant_manager = TenantManager()
    
    # Detect systems
    gpu_detector.detect_gpus()
    gpu_detector.create_port_mapping()
    
    # Create monitoring dashboard
    monitor = MonitoringDashboard(gpu_detector, tenant_manager)
    
    # Display static monitoring
    monitor.display_static_monitoring()
    
    # Show system health
    health = monitor.get_system_health()
    console.print(f"\n[bold]System Health: {health['health_percentage']:.1f}% ({health['status']})[/bold]")

if __name__ == "__main__":
    main()
