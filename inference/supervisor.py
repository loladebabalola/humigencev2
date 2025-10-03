# inference/supervisor.py

import os
import sys
import time
import signal
import subprocess
import threading
import yaml
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import psutil

console = Console()

def resolve_llama_binary(config_binary: str = None) -> str:
    """
    Resolve the llama.cpp inference binary.
    Priority:
    1. Explicit 'binary' from profile YAML
    2. Check for 'llama-server' in PATH
    3. Look for common binaries in ~/llama.cpp/build/bin/
    4. Check for 'llama-cli' in PATH as fallback
    """
    # 1. Profile-specified binary
    if config_binary and Path(config_binary).exists():
        console.print(f"[blue]🔧 Using profile-specified binary: {config_binary}[/blue]")
        return str(Path(config_binary).resolve())
    
    # 2. If llama-server is on PATH
    path_bin = shutil.which("llama-server")
    if path_bin:
        console.print(f"[blue]🔧 Using llama-server from PATH: {path_bin}[/blue]")
        return path_bin
    
    # 3. Check common llama.cpp build outputs
    home = Path.home()
    candidates = [
        home / "llama.cpp/build/bin/server",
        home / "llama.cpp/build/bin/llama-server",
        home / "llama.cpp/build/bin/llama-cli",
        home / "llama.cpp/server",
        home / "llama.cpp/llama-server",
        home / "llama.cpp/llama-cli",
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            console.print(f"[blue]🔧 Using llama binary from: {candidate}[/blue]")
            return str(candidate.resolve())
    
    # 4. Check for llama-cli in PATH as final fallback
    llama_cli = shutil.which("llama-cli")
    if llama_cli:
        console.print(f"[blue]🔧 Using llama-cli from PATH: {llama_cli}[/blue]")
        return llama_cli
    
    # 5. If nothing found, raise clear error
    raise FileNotFoundError(
        "❌ Could not locate a valid llama.cpp binary.\n"
        "Please either:\n"
        "1. Install llama.cpp and build the server binary\n"
        "2. Add llama-server to your PATH\n"
        "3. Set 'binary' field in your profile YAML\n"
        "4. Place llama.cpp binaries in ~/llama.cpp/build/bin/\n\n"
        "Searched locations:\n"
        f"  - PATH: llama-server, llama-cli\n"
        f"  - {home}/llama.cpp/build/bin/server\n"
        f"  - {home}/llama.cpp/build/bin/llama-server\n"
        f"  - {home}/llama.cpp/build/bin/llama-cli\n"
        f"  - {home}/llama.cpp/server\n"
        f"  - {home}/llama.cpp/llama-server\n"
        f"  - {home}/llama.cpp/llama-cli"
    )

class InferenceSupervisor:
    """Supervisor for managing multiple llama-server instances across GPUs"""
    
    def __init__(self, profile_path: str):
        self.profile_path = Path(profile_path)
        self.profile = self.load_profile()
        self.processes = {}
        self.running = False
        self.health_check_interval = 10  # seconds
        
    def load_profile(self) -> Dict[str, Any]:
        """Load the inference profile configuration"""
        try:
            with open(self.profile_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]❌ Failed to load profile: {e}[/red]")
            sys.exit(1)
    
    def start(self):
        """Start all llama-server instances"""
        console.print(f"[bold green]🚀 Starting Multi-Tenant Inference Supervisor[/bold green]")
        console.print(f"📋 Profile: {self.profile_path.name}")
        
        # Validate profile
        if not self.validate_profile():
            console.print("[red]❌ Invalid profile configuration![/red]")
            return False
        
        # Start instances
        instances = self.profile.get('instances', [])
        if not instances:
            console.print("[red]❌ No instances configured in profile![/red]")
            return False
        
        console.print(f"\n🔧 Starting {len(instances)} llama-server instance(s)...")
        
        success_count = 0
        for i, instance in enumerate(instances, 1):
            if self.start_instance(instance, i):
                success_count += 1
        
        if success_count == 0:
            console.print("[red]❌ Failed to start any instances![/red]")
            return False
        
        console.print(f"\n[green]✅ Started {success_count}/{len(instances)} instances successfully![/green]")
        
        # Start health monitoring
        self.running = True
        self.start_health_monitor()
        
        return True
    
    def validate_profile(self) -> bool:
        """Validate the profile configuration"""
        required_fields = ['model', 'instances']
        
        for field in required_fields:
            if field not in self.profile:
                console.print(f"[red]❌ Missing required field: {field}[/red]")
                return False
        
        # Validate model path
        model_path = self.profile['model']
        if not os.path.exists(model_path):
            console.print(f"[yellow]⚠️ Model file not found: {model_path}[/yellow]")
            console.print("This may cause issues when starting instances.")
        
        # Validate instances
        instances = self.profile['instances']
        if not isinstance(instances, list) or len(instances) == 0:
            console.print("[red]❌ No valid instances configured![/red]")
            return False
        
        for i, instance in enumerate(instances):
            required_instance_fields = ['gpu', 'port', 'parallel']
            for field in required_instance_fields:
                if field not in instance:
                    console.print(f"[red]❌ Instance {i+1} missing field: {field}[/red]")
                    return False
        
        return True
    
    def _filter_supported_args(self, cmd: List[str]) -> List[str]:
        """
        Filter out unsupported arguments to prevent launch errors.
        Keeps only the core arguments that are universally supported.
        """
        if not cmd:
            return cmd
            
        # The first element should always be the binary path
        binary_path = cmd[0]
        args = cmd[1:]
        
        # List of supported arguments
        supported_args = {
            '--model', '--port', '--ctx-size', '--host', '--parallel',
            '--n-gpu-layers', '--threads', '--gpu', '--batch-size',
            '--memory-f32', '--no-mmap', '--mlock', '--numa'
        }
        
        # Filter the arguments to keep only supported ones and their values
        filtered_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in supported_args:
                filtered_args.append(arg)
                # Add the value if it exists and isn't another argument
                if i + 1 < len(args) and not args[i + 1].startswith('--'):
                    filtered_args.append(args[i + 1])
                    i += 1
            elif not arg.startswith('--'):
                # This is a value, skip it (it was already added with its argument)
                pass
            else:
                # Unsupported argument, skip it and its value
                console.print(f"[yellow]⚠️ Skipping unsupported argument: {arg}[/yellow]")
                if i + 1 < len(args) and not args[i + 1].startswith('--'):
                    i += 1  # Skip the value too
            i += 1
        
        # Return binary path + filtered arguments
        return [binary_path] + filtered_args
    
    def start_instance(self, instance: Dict[str, Any], instance_num: int) -> bool:
        """Start a single llama-server instance"""
        gpu = instance['gpu']
        port = instance['port']
        parallel = instance['parallel']
        model_path = self.profile['model']
        context_window = self.profile.get('context_window', 131072)
        
        console.print(f"\n🔧 Starting Instance {instance_num}:")
        console.print(f"  GPU: {gpu}")
        console.print(f"  Port: {port}")
        console.print(f"  Parallel: {parallel}")
        
        # Check if port is already in use
        if self.is_port_in_use(port):
            console.print(f"[red]❌ Port {port} is already in use![/red]")
            return False
        
        # Resolve the llama binary
        try:
            binary = resolve_llama_binary(instance.get('binary'))
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return False
        
        # Verify binary exists and is executable
        if not Path(binary).exists():
            console.print(f"[red]❌ Binary not found: {binary}[/red]")
            return False
        
        if not os.access(binary, os.X_OK):
            console.print(f"[red]❌ Binary not executable: {binary}[/red]")
            return False
        
        # Build llama command with only supported arguments
        cmd = [
            binary,
            "--model", model_path,
            "--port", str(port),
            "--ctx-size", str(context_window),
            "--host", "0.0.0.0"
        ]
        
        # Add parallel processing if supported (some builds don't support this)
        if 'llama-cli' not in binary and 'llama-cli' not in Path(binary).name:
            cmd.extend(["--parallel", str(parallel)])
        
        # Add GPU layers for better performance
        cmd.extend(["--n-gpu-layers", "100"])
        
        # Add threads for CPU fallback
        cmd.extend(["--threads", "8"])
        
        # Filter out any unsupported arguments that might be in profile
        cmd = self._filter_supported_args(cmd)
        
        # Set CUDA_VISIBLE_DEVICES for this instance
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process info
            self.processes[instance_num] = {
                'process': process,
                'gpu': gpu,
                'port': port,
                'parallel': parallel,
                'start_time': time.time(),
                'restart_count': 0
            }
            
            # Wait a moment to check if it started successfully
            time.sleep(3)
            
            if process.poll() is None:
                console.print(f"[green]✅ Instance {instance_num} started successfully[/green]")
                return True
            else:
                stdout, stderr = process.communicate()
                console.print(f"[red]❌ Instance {instance_num} failed to start[/red]")
                console.print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Failed to start instance {instance_num}: {e}[/red]")
            return False
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def start_health_monitor(self):
        """Start the health monitoring thread"""
        def monitor():
            while self.running:
                try:
                    self.check_instance_health()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    console.print(f"[yellow]⚠️ Health monitor error: {e}[/yellow]")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        console.print("[blue]🔍 Health monitoring started[/blue]")
    
    def check_instance_health(self):
        """Check health of all instances and restart if needed"""
        for instance_num, instance_info in list(self.processes.items()):
            process = instance_info['process']
            
            if process.poll() is not None:
                # Process has died
                console.print(f"[yellow]⚠️ Instance {instance_num} (GPU {instance_info['gpu']}, Port {instance_info['port']}) has stopped[/yellow]")
                
                # Restart if restart count is below limit
                if instance_info['restart_count'] < 3:
                    console.print(f"[blue]🔄 Restarting instance {instance_num}...[/blue]")
                    
                    # Get the original instance config
                    instances = self.profile['instances']
                    if instance_num - 1 < len(instances):
                        original_instance = instances[instance_num - 1]
                        
                        # Remove old process
                        del self.processes[instance_num]
                        
                        # Start new instance
                        if self.start_instance(original_instance, instance_num):
                            self.processes[instance_num]['restart_count'] = instance_info['restart_count'] + 1
                            console.print(f"[green]✅ Instance {instance_num} restarted successfully[/green]")
                        else:
                            console.print(f"[red]❌ Failed to restart instance {instance_num}[/red]")
                    else:
                        console.print(f"[red]❌ Cannot restart instance {instance_num}: configuration not found[/red]")
                else:
                    console.print(f"[red]❌ Instance {instance_num} exceeded restart limit, giving up[/red]")
                    del self.processes[instance_num]
    
    def stop(self):
        """Stop all instances"""
        console.print("\n🛑 [bold]Stopping Multi-Tenant Inference Supervisor[/bold]")
        
        self.running = False
        
        if not self.processes:
            console.print("[yellow]⚠️ No instances running[/yellow]")
            return
        
        console.print(f"🛑 Stopping {len(self.processes)} instance(s)...")
        
        for instance_num, instance_info in self.processes.items():
            process = instance_info['process']
            gpu = instance_info['gpu']
            port = instance_info['port']
            
            console.print(f"🛑 Stopping instance {instance_num} (GPU {gpu}, Port {port})...")
            
            try:
                # Try graceful shutdown first
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    console.print(f"[green]✅ Instance {instance_num} stopped gracefully[/green]")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    process.kill()
                    process.wait()
                    console.print(f"[yellow]⚠️ Instance {instance_num} force killed[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]❌ Error stopping instance {instance_num}: {e}[/red]")
        
        self.processes.clear()
        console.print("[green]✅ All instances stopped[/green]")
    
    def status(self):
        """Show status of all instances"""
        console.print("\n📊 [bold]Instance Status[/bold]")
        console.print("="*50)
        
        if not self.processes:
            console.print("[yellow]⚠️ No instances running[/yellow]")
            return
        
        table = Table(show_header=True, box=None)
        table.add_column("Instance", style="cyan", width=8)
        table.add_column("GPU", style="white", width=4)
        table.add_column("Port", style="green", width=6)
        table.add_column("Parallel", style="blue", width=8)
        table.add_column("Status", style="white", width=12)
        table.add_column("Uptime", style="yellow", width=12)
        table.add_column("Restarts", style="red", width=8)
        
        for instance_num, instance_info in self.processes.items():
            process = instance_info['process']
            gpu = instance_info['gpu']
            port = instance_info['port']
            parallel = instance_info['parallel']
            start_time = instance_info['start_time']
            restart_count = instance_info['restart_count']
            
            # Check if process is running
            if process.poll() is None:
                status = "🟢 Running"
                uptime = time.time() - start_time
                uptime_str = f"{uptime:.0f}s"
            else:
                status = "🔴 Stopped"
                uptime_str = "N/A"
            
            table.add_row(
                str(instance_num),
                str(gpu),
                str(port),
                str(parallel),
                status,
                uptime_str,
                str(restart_count)
            )
        
        console.print(table)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print(f"\n[yellow]⚠️ Received signal {signum}, shutting down...[/yellow]")
        self.stop()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Multi-Tenant Inference Supervisor")
    parser.add_argument("--profile", required=True, help="Path to profile YAML file")
    parser.add_argument("--action", choices=["start", "stop", "status"], default="start", help="Action to perform")
    
    args = parser.parse_args()
    
    # Create supervisor
    supervisor = InferenceSupervisor(args.profile)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, supervisor.signal_handler)
    signal.signal(signal.SIGTERM, supervisor.signal_handler)
    
    if args.action == "start":
        if supervisor.start():
            try:
                # Keep running until interrupted
                while supervisor.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                supervisor.signal_handler(signal.SIGINT, None)
    elif args.action == "stop":
        supervisor.stop()
    elif args.action == "status":
        supervisor.status()

if __name__ == "__main__":
    main()
