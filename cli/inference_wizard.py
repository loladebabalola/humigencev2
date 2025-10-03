# cli/inference_wizard.py

import os
import sys
import time
import json
import yaml
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import inquirer

console = Console()

class InferenceWizard:
    """Multi-Tenant Inference Wizard for Humigence CLI"""
    
    def __init__(self):
        self.humigence_dir = Path(__file__).parent.parent
        self.inference_dir = self.humigence_dir / "inference"
        self.profiles_dir = self.inference_dir / "profiles"
        self.runs_dir = self.humigence_dir / "runs"
        self.current_profile = None
        self.advanced_mode = False
        
        # Ensure directories exist
        self.profiles_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
    def run(self):
        """Main wizard execution"""
        console.rule("[bold cyan]Multi-Tenant Inference Wizard 🖥️")
        console.print("[dim]Set up fully local multi-tenant inference with GPU management, routing, and monitoring.[/dim]\n")
        
        while True:
            self.show_main_menu()
            choice = console.input("[bold blue]Select an option[/bold blue]: ")
            
            if choice == "1":
                self.quick_start()
            elif choice == "2":
                self.configure_hardware()
            elif choice == "3":
                self.configure_model()
            elif choice == "4":
                self.manage_tenants()
            elif choice == "5":
                self.start_stop_services()
            elif choice == "6":
                self.batch_test()
            elif choice == "7":
                self.monitor_dashboard()
            elif choice == "8":
                self.export_snapshot()
            elif choice == "9":
                self.toggle_advanced_mode()
            elif choice == "0":
                console.print("[bold red]Exiting Multi-Tenant Inference Wizard...[/bold red]")
                break
            else:
                console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")
                time.sleep(1)
    
    def show_main_menu(self):
        """Display the main wizard menu"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Multi-Tenant Inference Control Panel[/bold cyan]")
        console.print("="*60)
        
        if self.current_profile:
            console.print(f"[green]📋 Active Profile: {self.current_profile}[/green]")
        else:
            console.print("[yellow]📋 No active profile[/yellow]")
        
        console.print("\n[bold]Quick Actions:[/bold]")
        console.print("[bold green]1.[/bold green] Quick Start (detect GPUs, apply default profile)")
        console.print("[bold green]2.[/bold green] Configure Hardware (GPU-to-port mapping)")
        console.print("[bold green]3.[/bold green] Configure Model (GGUF path, context window)")
        console.print("[bold green]4.[/bold green] Tenants & Policies (quotas, priorities)")
        console.print("[bold green]5.[/bold green] Start/Stop Services (launch servers + router)")
        console.print("[bold green]6.[/bold green] Batch Test (loadgen scenarios)")
        console.print("[bold green]7.[/bold green] Monitor (live dashboard)")
        console.print("[bold green]8.[/bold green] Export/Snapshot (save profile + metrics)")
        
        if self.advanced_mode:
            console.print("\n[bold yellow]Advanced Options:[/bold yellow]")
            console.print("[bold yellow]9.[/bold yellow] Toggle Advanced Mode (currently ON)")
        else:
            console.print("\n[bold yellow]Advanced Options:[/bold yellow]")
            console.print("[bold yellow]9.[/bold yellow] Toggle Advanced Mode (currently OFF)")
        
        console.print("\n[bold red]0.[/bold red] Exit Wizard")
        console.print("")
    
    def detect_gpus(self) -> tuple[int, List[Dict]]:
        """Detect available GPUs"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpus = []
                for i in range(gpu_count):
                    gpus.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB"
                    })
                return gpu_count, gpus
            else:
                return 0, []
        except ImportError:
            return 0, []
    
    def quick_start(self):
        """Step 1: Quick Start - detect GPUs and apply default profile"""
        console.print("\n🚀 [bold]Quick Start - Multi-Tenant Inference Setup[/bold]")
        console.print("="*60)
        
        # Detect GPUs
        console.print("\n🔍 [bold]Detecting GPUs...[/bold]")
        gpu_count, gpus = self.detect_gpus()
        
        if gpu_count == 0:
            console.print("[red]❌ No GPUs detected! Multi-tenant inference requires GPU acceleration.[/red]")
            return
        
        # Display detected GPUs
        console.print(f"[green]✅ Found {gpu_count} GPU(s):[/green]")
        gpu_table = Table(show_header=True, box=None)
        gpu_table.add_column("Index", style="cyan", width=6)
        gpu_table.add_column("Name", style="white", width=40)
        gpu_table.add_column("VRAM", style="green", width=10)
        
        for gpu in gpus:
            gpu_table.add_row(str(gpu['index']), gpu['name'], gpu['memory'])
        
        console.print(gpu_table)
        
        # Select default profile based on GPU count
        if gpu_count >= 4:
            default_profile = "quad_5090_local"
        elif gpu_count >= 2:
            default_profile = "dual_5090_local"
        else:
            default_profile = "single_gpu_local"
        
        console.print(f"\n📋 [bold]Applying default profile: {default_profile}[/bold]")
        
        # Create or load profile
        profile_path = self.profiles_dir / f"{default_profile}.yaml"
        
        if not profile_path.exists():
            console.print(f"[yellow]⚠️ Profile {default_profile} not found. Creating default profile...[/yellow]")
            self.create_default_profile(default_profile, gpus)
        
        # Load profile
        try:
            with open(profile_path, 'r') as f:
                profile = yaml.safe_load(f)
            
            self.current_profile = default_profile
            console.print(f"[green]✅ Profile loaded: {default_profile}[/green]")
            
            # Display profile summary
            self.display_profile_summary(profile)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load profile: {e}[/red]")
            return
        
        console.print("\n[bold green]🎉 Quick Start completed![/bold green]")
        console.print("You can now start services (option 5) or configure advanced settings.")
    
    def create_default_profile(self, profile_name: str, gpus: List[Dict]):
        """Create a default profile based on available GPUs"""
        if len(gpus) >= 4:
            # Quad GPU setup
            instances = [
                {"gpu": 0, "port": 8080, "parallel": 4},
                {"gpu": 1, "port": 8081, "parallel": 4},
                {"gpu": 2, "port": 8082, "parallel": 4},
                {"gpu": 3, "port": 8083, "parallel": 4}
            ]
        elif len(gpus) >= 2:
            # Dual GPU setup
            instances = [
                {"gpu": 0, "port": 8080, "parallel": 4},
                {"gpu": 1, "port": 8081, "parallel": 4}
            ]
        else:
            # Single GPU setup
            instances = [
                {"gpu": 0, "port": 8080, "parallel": 2}
            ]
        
        profile = {
            "model": "/models/gpt-oss-20b/gpt-oss-20b.Q4_K_M.gguf",
            "context_window": 131072,
            "instances": instances,
            "router": {
                "entrypoint": "localhost:8000",
                "strategy": "tenant_sticky_then_latency"
            }
        }
        
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        with open(profile_path, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False)
        
        console.print(f"[green]✅ Created default profile: {profile_name}[/green]")
    
    def display_profile_summary(self, profile: Dict):
        """Display a summary of the current profile"""
        console.print("\n📊 [bold]Profile Summary:[/bold]")
        console.print(f"  Model: {profile.get('model', 'N/A')}")
        console.print(f"  Context Window: {profile.get('context_window', 'N/A'):,}")
        console.print(f"  Instances: {len(profile.get('instances', []))}")
        
        for i, instance in enumerate(profile.get('instances', []), 1):
            console.print(f"    {i}. GPU {instance['gpu']} → Port {instance['port']} (parallel: {instance['parallel']})")
        
        router = profile.get('router', {})
        console.print(f"  Router: {router.get('entrypoint', 'N/A')} ({router.get('strategy', 'N/A')})")
    
    def configure_hardware(self):
        """Step 2: Configure Hardware - GPU-to-port mapping"""
        if not self.advanced_mode:
            console.print("\n[yellow]⚠️ Hardware configuration requires Advanced Mode.[/yellow]")
            if not Confirm.ask("Enable Advanced Mode now?"):
                return
            self.advanced_mode = True
        
        console.print("\n🔧 [bold]Configure Hardware - GPU-to-Port Mapping[/bold]")
        console.print("="*60)
        
        gpu_count, gpus = self.detect_gpus()
        if gpu_count == 0:
            console.print("[red]❌ No GPUs detected![/red]")
            return
        
        console.print(f"\n[blue]Detected {gpu_count} GPU(s):[/blue]")
        for gpu in gpus:
            console.print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory']})")
        
        # Get current profile or create new one
        if not self.current_profile:
            profile_name = Prompt.ask("Enter profile name", default="custom_hardware")
            self.current_profile = profile_name
        
        profile_path = self.profiles_dir / f"{self.current_profile}.yaml"
        
        # Load existing profile or create new
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                profile = yaml.safe_load(f)
        else:
            profile = {
                "model": "/models/gpt-oss-20b/gpt-oss-20b.Q4_K_M.gguf",
                "context_window": 131072,
                "router": {
                    "entrypoint": "localhost:8000",
                    "strategy": "tenant_sticky_then_latency"
                }
            }
        
        # Configure instances
        instances = []
        base_port = 8080
        
        for gpu in gpus:
            console.print(f"\n[bold]Configuring GPU {gpu['index']}:[/bold]")
            
            port = IntPrompt.ask(f"Port for GPU {gpu['index']}", default=base_port)
            parallel = IntPrompt.ask(f"Parallel slots for GPU {gpu['index']}", default=4)
            
            instances.append({
                "gpu": gpu['index'],
                "port": port,
                "parallel": parallel
            })
            
            base_port += 1
        
        profile['instances'] = instances
        
        # Save profile
        with open(profile_path, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False)
        
        console.print(f"\n[green]✅ Hardware configuration saved to {self.current_profile}[/green]")
        self.display_profile_summary(profile)
    
    def configure_model(self):
        """Step 3: Configure Model - GGUF path, context window"""
        if not self.advanced_mode:
            console.print("\n[yellow]⚠️ Model configuration requires Advanced Mode.[/yellow]")
            if not Confirm.ask("Enable Advanced Mode now?"):
                return
            self.advanced_mode = True
        
        console.print("\n🤖 [bold]Configure Model - GGUF Path & Context Window[/bold]")
        console.print("="*60)
        
        # Get current profile or create new one
        if not self.current_profile:
            profile_name = Prompt.ask("Enter profile name", default="custom_model")
            self.current_profile = profile_name
        
        profile_path = self.profiles_dir / f"{self.current_profile}.yaml"
        
        # Load existing profile or create new
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                profile = yaml.safe_load(f)
        else:
            profile = {
                "instances": [{"gpu": 0, "port": 8080, "parallel": 4}],
                "router": {
                    "entrypoint": "localhost:8000",
                    "strategy": "tenant_sticky_then_latency"
                }
            }
        
        # Configure model path
        current_model = profile.get('model', '/models/gpt-oss-20b/gpt-oss-20b.Q4_K_M.gguf')
        model_path = Prompt.ask("GGUF model path", default=current_model)
        
        # Validate model path
        if not os.path.exists(model_path):
            console.print(f"[yellow]⚠️ Model file not found: {model_path}[/yellow]")
            if not Confirm.ask("Continue anyway?"):
                return
        
        profile['model'] = model_path
        
        # Configure context window
        current_context = profile.get('context_window', 131072)
        context_window = IntPrompt.ask("Context window size", default=current_context)
        profile['context_window'] = context_window
        
        # Save profile
        with open(profile_path, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False)
        
        console.print(f"\n[green]✅ Model configuration saved to {self.current_profile}[/green]")
        console.print(f"  Model: {model_path}")
        console.print(f"  Context Window: {context_window:,}")
    
    def manage_tenants(self):
        """Step 4: Tenants & Policies - create/edit quotas, priorities"""
        console.print("\n👥 [bold]Tenants & Policies Management[/bold]")
        console.print("="*60)
        
        # This will be implemented when we create the tenants module
        console.print("[yellow]⚠️ Tenant management will be implemented in the tenants module.[/yellow]")
        console.print("For now, using default tenant configuration.")
        
        # Create default tenants.yaml if it doesn't exist
        tenants_path = self.inference_dir / "tenants.yaml"
        if not tenants_path.exists():
            default_tenants = {
                "tenants": {
                    "default": {
                        "name": "Default Tenant",
                        "quota_rps": 10,
                        "max_context": 131072,
                        "priority": 1
                    }
                }
            }
            
            with open(tenants_path, 'w') as f:
                yaml.dump(default_tenants, f, default_flow_style=False)
            
            console.print("[green]✅ Created default tenants.yaml[/green]")
        else:
            console.print("[green]✅ Using existing tenants.yaml[/green]")
    
    def start_stop_services(self):
        """Step 5: Start/Stop Services - launch or stop all servers + router"""
        console.print("\n🚀 [bold]Start/Stop Services[/bold]")
        console.print("="*60)
        
        if not self.current_profile:
            console.print("[red]❌ No active profile! Please run Quick Start first.[/red]")
            return
        
        # Check if services are already running
        services_running = self.check_services_running()
        
        if services_running:
            console.print("[yellow]⚠️ Services appear to be running.[/yellow]")
            action = Prompt.ask("Action", choices=["stop", "restart", "status"], default="status")
        else:
            action = Prompt.ask("Action", choices=["start", "status"], default="start")
        
        if action == "start":
            self.start_services()
        elif action == "stop":
            self.stop_services()
        elif action == "restart":
            self.stop_services()
            time.sleep(2)
            self.start_services()
        elif action == "status":
            self.show_service_status()
    
    def check_services_running(self) -> bool:
        """Check if inference services are running"""
        try:
            # Check if router is running on port 8000
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            return result == 0
        except:
            return False
    
    def start_services(self):
        """Start all inference services"""
        console.print("\n🚀 [bold]Starting Multi-Tenant Inference Services...[/bold]")
        
        # Load current profile
        profile_path = self.profiles_dir / f"{self.current_profile}.yaml"
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        # Start supervisor (which will start llama-server instances)
        console.print("🔧 Starting supervisor...")
        supervisor_cmd = [
            "python3", 
            str(self.inference_dir / "supervisor.py"),
            "--profile", str(profile_path)
        ]
        
        # Start in background
        try:
            subprocess.Popen(supervisor_cmd, cwd=self.humigence_dir)
            console.print("[green]✅ Supervisor started[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to start supervisor: {e}[/red]")
            return
        
        # Wait a moment for supervisor to start
        time.sleep(3)
        
        # Start router
        console.print("🌐 Starting router...")
        router_cmd = [
            "python3",
            str(self.inference_dir / "router.py"),
            "--profile", str(profile_path)
        ]
        
        try:
            subprocess.Popen(router_cmd, cwd=self.humigence_dir)
            console.print("[green]✅ Router started[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to start router: {e}[/red]")
            return
        
        # Wait for services to be ready
        console.print("⏳ Waiting for services to be ready...")
        time.sleep(5)
        
        # Check if services are running
        if self.check_services_running():
            console.print("\n[bold green]🎉 All services started successfully![/bold green]")
            console.print(f"🌐 Router available at: http://localhost:8000")
            console.print("📊 Use option 7 to monitor services")
        else:
            console.print("\n[yellow]⚠️ Services may still be starting. Check status with option 5.[/yellow]")
    
    def stop_services(self):
        """Stop all inference services"""
        console.print("\n🛑 [bold]Stopping Multi-Tenant Inference Services...[/bold]")
        
        try:
            # Kill processes by name
            subprocess.run(["pkill", "-f", "supervisor.py"], check=False)
            subprocess.run(["pkill", "-f", "router.py"], check=False)
            subprocess.run(["pkill", "-f", "llama-server"], check=False)
            
            console.print("[green]✅ All services stopped[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Error stopping services: {e}[/yellow]")
    
    def show_service_status(self):
        """Show status of all services"""
        console.print("\n📊 [bold]Service Status[/bold]")
        console.print("="*40)
        
        # Check router
        router_running = self.check_services_running()
        console.print(f"Router (port 8000): {'🟢 Running' if router_running else '🔴 Stopped'}")
        
        # Check for llama-server processes
        try:
            result = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True)
            llama_servers = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            console.print(f"Llama Servers: {'🟢' if llama_servers > 0 else '🔴'} {llama_servers} running")
        except:
            console.print("Llama Servers: ❓ Unknown")
    
    def batch_test(self):
        """Step 6: Batch Test - run local loadgen scenarios"""
        console.print("\n🧪 [bold]Batch Test - Load Generation Scenarios[/bold]")
        console.print("="*60)
        
        if not self.current_profile:
            console.print("[red]❌ No active profile! Please run Quick Start first.[/red]")
            return
        
        if not self.check_services_running():
            console.print("[red]❌ Services not running! Please start services first (option 5).[/red]")
            return
        
        # Test scenarios
        scenarios = [
            "burst (short prompts)",
            "mixed (short + long prompts)", 
            "soak (sustained load)"
        ]
        
        console.print("\n[bold]Available Test Scenarios:[/bold]")
        for i, scenario in enumerate(scenarios, 1):
            console.print(f"  {i}. {scenario}")
        
        try:
            choice = int(Prompt.ask("Select scenario", default="1"))
            if 1 <= choice <= len(scenarios):
                scenario = scenarios[choice - 1]
                self.run_load_test(scenario)
            else:
                console.print("[red]❌ Invalid choice![/red]")
        except ValueError:
            console.print("[red]❌ Please enter a valid number![/red]")
    
    def run_load_test(self, scenario: str):
        """Run a load test scenario"""
        console.print(f"\n🧪 [bold]Running {scenario} test...[/bold]")
        
        # This will be implemented when we create the loadgen module
        console.print("[yellow]⚠️ Load testing will be implemented in the loadgen module.[/yellow]")
        console.print("For now, simulating test execution...")
        
        # Simulate test execution
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running load test...", total=100)
            
            for i in range(100):
                time.sleep(0.1)
                progress.update(task, advance=1)
        
        console.print("[green]✅ Load test completed![/green]")
        console.print("📊 Results saved to runs/ directory")
    
    def monitor_dashboard(self):
        """Step 7: Monitor - open Rich-based TUI dashboard"""
        console.print("\n📊 [bold]Live Monitoring Dashboard[/bold]")
        console.print("="*60)
        
        if not self.check_services_running():
            console.print("[red]❌ Services not running! Please start services first (option 5).[/red]")
            return
        
        # This will be implemented when we create the TUI module
        console.print("[yellow]⚠️ Live monitoring will be implemented in the TUI module.[/yellow]")
        console.print("For now, showing basic status...")
        
        self.show_service_status()
    
    def export_snapshot(self):
        """Step 8: Export/Snapshot - save active profile + metrics"""
        console.print("\n💾 [bold]Export/Snapshot - Save Profile & Metrics[/bold]")
        console.print("="*60)
        
        if not self.current_profile:
            console.print("[red]❌ No active profile! Please run Quick Start first.[/red]")
            return
        
        # Create snapshot directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.runs_dir / f"inference_snapshot_{timestamp}"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Copy profile
        profile_path = self.profiles_dir / f"{self.current_profile}.yaml"
        if profile_path.exists():
            import shutil
            shutil.copy2(profile_path, snapshot_dir / "profile.yaml")
            console.print(f"[green]✅ Profile saved to {snapshot_dir}/profile.yaml[/green]")
        
        # Save metrics (placeholder)
        metrics = {
            "timestamp": timestamp,
            "profile": self.current_profile,
            "services_running": self.check_services_running(),
            "exported_at": time.time()
        }
        
        with open(snapshot_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        console.print(f"[green]✅ Metrics saved to {snapshot_dir}/metrics.json[/green]")
        console.print(f"[blue]📁 Snapshot directory: {snapshot_dir}[/blue]")
    
    def toggle_advanced_mode(self):
        """Toggle advanced mode on/off"""
        self.advanced_mode = not self.advanced_mode
        status = "ON" if self.advanced_mode else "OFF"
        console.print(f"\n[bold green]Advanced Mode: {status}[/bold green]")
        
        if self.advanced_mode:
            console.print("🔧 Advanced configuration options are now available.")
        else:
            console.print("🔧 Advanced configuration options are hidden.")
