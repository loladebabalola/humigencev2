# cli/multi_tenant.py

import os
import sys
import time
import json
import subprocess
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich import box

# Add inference directory to path
sys.path.append(str(Path(__file__).parent.parent / "inference"))

from gpu_detector import GPUDetector
from model_detector import ModelDetector
from tenant_manager import TenantManager
from summary_panel import SummaryPanel
from profile_manager import ProfileManager

console = Console()

def generate_profile_yaml(model_path: str, context_length: int, gpu_ports: List[int], tenants: Optional[Dict] = None, binary_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a complete, valid profile YAML for the supervisor.
    Ensures all required fields are present to prevent "Missing required field" errors.
    Always creates a valid profile with proper fallbacks.
    """
    PROFILE_DIR = Path.home() / "humigence/inference/profiles"
    PROFILE_FILE = PROFILE_DIR / "auto_detected.yaml"
    
    # Ensure directory exists
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate and provide fallbacks for required fields
    if not model_path or not str(model_path).strip():
        model_path = "/home/joshua/models/placeholder.gguf"
        console.print("[yellow]⚠️ No model path provided, using placeholder[/yellow]")
    
    if not context_length or context_length <= 0:
        context_length = 131072
        console.print("[yellow]⚠️ Invalid context length, using default: 131072[/yellow]")
    
    if not gpu_ports or len(gpu_ports) == 0:
        gpu_ports = [8000, 8001]  # Default ports
        console.print("[yellow]⚠️ No GPU ports provided, using default: [8000, 8001][/yellow]")
    
    # Build instances from GPU ports
    instances = []
    for gpu, port in enumerate(gpu_ports):
        instance = {
            "gpu": gpu,
            "port": port,
            "parallel": 4  # Default parallel slots
        }
        # Add binary path if specified
        if binary_path:
            instance["binary"] = binary_path
        instances.append(instance)
    
    # Ensure at least one tenant exists
    if not tenants:
        tenants = {
            "tenant_default": {
                "quota": "100%",
                "priority": "normal",
                "endpoint": f"http://localhost:{gpu_ports[0] if gpu_ports else 8000}",
                "api_key": None
            }
        }
        console.print("[yellow]⚠️ No tenants provided, creating default tenant[/yellow]")
    
    # Create the profile structure that supervisor expects
    profile = {
        "profile": "auto_detected",
        "model": str(model_path),
        "context_length": int(context_length),
        "instances": instances,
        "router": {
            "strategy": "round_robin",
            "entrypoint": "localhost:8000"
        },
        "tenants": tenants
    }
    
    # Write the profile YAML
    try:
        with PROFILE_FILE.open("w") as f:
            yaml.safe_dump(profile, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]✅ Profile written: {PROFILE_FILE}[/green]")
        return profile
        
    except Exception as e:
        console.print(f"[red]❌ Failed to write profile: {e}[/red]")
        return {}

def detect_llama_binary() -> Optional[str]:
    """
    Detect available llama.cpp binary during auto-setup.
    Returns the path to the binary if found, None otherwise.
    """
    import shutil
    from pathlib import Path
    
    # Check PATH first
    path_bin = shutil.which("llama-server")
    if path_bin:
        return path_bin
    
    # Check common build locations
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
            return str(candidate.resolve())
    
    # Check for llama-cli in PATH as fallback
    llama_cli = shutil.which("llama-cli")
    if llama_cli:
        return llama_cli
    
    return None

class MultiTenantWizard:
    """Zero-friction Multi-Tenant Inference Wizard"""
    
    def __init__(self):
        self.humigence_dir = Path(__file__).parent.parent
        self.inference_dir = self.humigence_dir / "inference"
        self.profiles_dir = self.inference_dir / "profiles"
        
        # Initialize components
        self.gpu_detector = GPUDetector()
        self.model_detector = ModelDetector()
        self.tenant_manager = TenantManager()
        self.profile_manager = ProfileManager(str(self.profiles_dir))
        self.summary_panel = SummaryPanel(
            self.gpu_detector, 
            self.model_detector, 
            self.tenant_manager
        )
        
        # Current state
        self.current_profile = None
        self.services_running = False
        
        # Ensure directories exist
        self.profiles_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Main wizard execution - zero-friction flow"""
        console.rule("[bold cyan]Multi-Tenant Inference Wizard 🖥️[/bold cyan]")
        console.print("[dim]Set up fully local multi-tenant inference with GPU management, routing, and monitoring.[/dim]\n")
        
        # Auto-detect and setup everything
        self.auto_setup()
        
        # Show main control panel
        while True:
            self.show_control_panel()
            choice = console.input("\n[bold blue]Select an option[/bold blue]: ")
            
            if choice == "1":
                self.start_stop_services()
            elif choice == "2":
                self.monitor_dashboard()
            elif choice == "3":
                self.tenant_manager_menu()
            elif choice == "4":
                self.advanced_options()
            elif choice == "0":
                console.print("[bold red]Exiting Multi-Tenant Inference Wizard...[/bold red]")
                break
            else:
                console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")
                time.sleep(1)
    
    def auto_setup(self):
        """Auto-detect GPUs, models, and create default tenant - zero friction"""
        console.print("[bold green]🚀 Auto-Setup: Zero-Friction Configuration[/bold green]")
        console.print("="*60)
        
        # Step 1: Detect GPUs
        console.print("🔍 Detecting GPUs...")
        gpu_count, gpus = self.gpu_detector.detect_gpus()
        if gpu_count > 0:
            self.gpu_detector.create_port_mapping()
            console.print(f"✅ Detected {gpu_count} GPU(s) and mapped to ports")
        else:
            console.print("❌ No GPUs detected! Multi-tenant inference requires GPU acceleration.")
            return
        
        # Step 2: Detect Models
        console.print("🤖 Scanning for models...")
        model_count, models = self.model_detector.scan_models()
        if model_count > 0:
            console.print(f"✅ Found {model_count} model(s)")
        else:
            console.print("⚠️ No models found. Please add GGUF models to ~/models/ directory.")
            # Continue anyway with default config
        
        # Step 3: Create Default Tenant
        console.print("👥 Setting up default tenant...")
        if not self.tenant_manager.tenants:
            self.tenant_manager.create_default_tenant()
        console.print("✅ Default tenant configured")
        
        # Step 4: Create Profile
        console.print("📁 Creating profile...")
        self.create_auto_profile()
        console.print("✅ Profile created")
        
        # Show summary
        self.show_auto_setup_summary()
    
    def create_auto_profile(self):
        """Create profile automatically based on detected hardware"""
        console.print("🔧 Building complete profile with all required fields...")
        
        # Get detected information
        gpu_summary = self.gpu_detector.get_system_summary()
        model_summary = self.model_detector.get_system_summary()
        tenant_summary = self.tenant_manager.get_system_summary()
        
        # Extract required information with robust fallbacks
        gpu_ports = list(gpu_summary["port_mapping"].values()) if gpu_summary.get("port_mapping") else [8000, 8001]
        
        # Get model path and context length with validation
        model_path = None
        context_length = None
        
        if model_summary.get("default_model"):
            model_path = model_summary["default_model"].get("path")
            context_length = model_summary["default_model"].get("context_window")
            
            # Validate context_length
            if not isinstance(context_length, (int, float)) or context_length <= 0:
                context_length = 131072
                console.print("[yellow]⚠️ Invalid context length detected, using default: 131072[/yellow]")
        else:
            # Fallback if no model detected
            model_path = "/home/joshua/models/placeholder.gguf"
            context_length = 131072
            console.print("[yellow]⚠️ No model detected, using placeholder path[/yellow]")
        
        # Convert tenant summary to the format expected by generate_profile_yaml
        tenants = {}
        if tenant_summary.get("tenants"):
            for tenant in tenant_summary["tenants"]:
                tenants[tenant["alias"]] = {
                    "quota": f"{tenant.get('quota_percentage', 100)}%",
                    "priority": tenant.get("priority", "normal"),
                    "endpoint": tenant.get("endpoint", f"http://localhost:{gpu_ports[0] if gpu_ports else 8000}"),
                    "api_key": None
                }
        
        # Detect llama binary
        console.print("🔍 Detecting llama.cpp binary...")
        binary_path = detect_llama_binary()
        if binary_path:
            console.print(f"✅ Found llama binary: {binary_path}")
        else:
            console.print("[yellow]⚠️ No llama binary detected, will auto-detect at startup[/yellow]")
        
        # Generate the complete profile YAML with all fallbacks
        profile = generate_profile_yaml(
            model_path=model_path,
            context_length=context_length,
            gpu_ports=gpu_ports,
            tenants=tenants if tenants else None,
            binary_path=binary_path
        )
        
        if profile:
            self.current_profile = "auto_detected"
            console.print("[green]✅ Profile created with all required fields[/green]")
        else:
            console.print("[red]❌ Failed to create auto profile[/red]")
    
    def show_auto_setup_summary(self):
        """Show summary after auto-setup"""
        console.print("\n[bold green]🎉 Auto-Setup Complete![/bold green]")
        console.print("="*50)
        
        # Get summaries
        gpu_summary = self.gpu_detector.get_system_summary()
        model_summary = self.model_detector.get_system_summary()
        tenant_summary = self.tenant_manager.get_system_summary()
        
        # Display summary
        console.print(f"[blue]📊 Summary:[/blue]")
        console.print(f"• GPUs detected: {gpu_summary['gpu_count']} (mapped to ports {list(gpu_summary['port_mapping'].values())})")
        
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
        
        console.print("• Status: STOPPED")
        
        console.print("\n[bold]Quick Actions:[/bold]")
        console.print("1. Start/Stop Services")
        console.print("2. Monitor (live dashboard)")
        console.print("3. Tenant Manager (add/remove/list aliases)")
        console.print("4. Advanced Options")
        console.print("0. Exit Wizard")
    
    def show_control_panel(self):
        """Display the main control panel"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Multi-Tenant Inference Control Panel[/bold cyan]")
        console.print("="*60)
        
        # Get current status
        service_status = self.summary_panel.check_service_status()
        health = self.summary_panel.get_system_health()
        
        # Display status with guidance hints
        if service_status["router"]:
            router_status = "🟢 Running"
        else:
            router_status = "🔴 Stopped (press 1 to Start Services)"
        
        if service_status["llama_servers"] > 0:
            llama_status = f"🟢 {service_status['llama_servers']}/{service_status['total_instances']} running"
        else:
            llama_status = "🔴 0/0 (waiting for startup)"
        
        console.print(f"[green]📋 Active Profile: {self.current_profile or 'None'}[/green]")
        console.print(f"[blue]🌐 Router: {router_status}[/blue]")
        console.print(f"[blue]🤖 Llama Servers: {llama_status}[/blue]")
        
        # Show real GPU health if available
        if health.get('gpu_health_info'):
            gpu_info = health['gpu_health_info']
            temp_str = f", {gpu_info['temperature_c']:.0f}°C" if gpu_info.get('temperature_c') else ""
            console.print(f"[blue]💚 GPU Health: {gpu_info['utilization_percent']:.0f}% util, {gpu_info['memory_used_gb']:.1f}GB/{gpu_info['memory_total_gb']:.1f}GB VRAM{temp_str}[/blue]")
        else:
            console.print(f"[blue]💚 System Health: {health['health_percentage']:.1f}% ({health['status']})[/blue]")
        
        console.print("\n[bold]Quick Actions:[/bold]")
        console.print("[bold green]1.[/bold green] Start/Stop Services")
        console.print("[bold green]2.[/bold green] Monitor (live dashboard)")
        console.print("[bold green]3.[/bold green] Tenant Manager (add/remove/list aliases)")
        console.print("[bold green]4.[/bold green] Advanced Options")
        console.print("\n[bold red]0.[/bold red] Exit Wizard")
    
    def start_stop_services(self):
        """Start/Stop Services - launch or stop all servers + router"""
        console.print("\n🚀 [bold]Start/Stop Services[/bold]")
        console.print("="*60)
        
        if not self.current_profile:
            console.print("[red]❌ No active profile! Please run auto-setup first.[/red]")
            return
        
        # Check current status
        service_status = self.summary_panel.check_service_status()
        
        if service_status["router"] or service_status["llama_servers"] > 0:
            console.print("[yellow]⚠️ Services are currently running.[/yellow]")
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
    
    def start_services(self):
        """Start all inference services"""
        console.print("\n🚀 [bold]Starting Multi-Tenant Inference Services...[/bold]")
        
        # Load current profile
        profile_data = self.profile_manager.load_profile(self.current_profile)
        if not profile_data:
            console.print("[red]❌ Failed to load profile![/red]")
            return
        
        # Stop any existing services first
        console.print("🛑 Stopping existing services...")
        self.stop_services()
        time.sleep(2)  # Give services time to stop
        
        # Start supervisor (which will start llama-server instances)
        console.print("🔧 Starting supervisor...")
        supervisor_cmd = [
            "python3", 
            str(self.inference_dir / "supervisor.py"),
            "--profile", str(self.profiles_dir / f"{self.current_profile}.yaml")
        ]
        
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
            "--profile", str(self.profiles_dir / f"{self.current_profile}.yaml")
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
        service_status = self.summary_panel.check_service_status()
        if service_status["router"]:
            console.print("\n[bold green]🎉 All services started successfully![/bold green]")
            console.print(f"🌐 Router available at: http://localhost:8000")
            console.print("📊 Use option 2 to monitor services")
            self.services_running = True
        else:
            console.print("\n[yellow]⚠️ Services may still be starting. Check status with option 1.[/yellow]")
    
    def stop_services(self):
        """Stop all inference services"""
        console.print("\n🛑 [bold]Stopping Multi-Tenant Inference Services...[/bold]")
        
        try:
            # Kill processes by name
            subprocess.run(["pkill", "-f", "supervisor.py"], check=False)
            subprocess.run(["pkill", "-f", "router.py"], check=False)
            subprocess.run(["pkill", "-f", "llama-server"], check=False)
            
            # Also kill any processes using our ports
            subprocess.run(["fuser", "-k", "8000/tcp"], check=False)
            subprocess.run(["fuser", "-k", "8001/tcp"], check=False)
            
            console.print("[green]✅ All services stopped[/green]")
            self.services_running = False
        except Exception as e:
            console.print(f"[yellow]⚠️ Error stopping services: {e}[/yellow]")
    
    def show_service_status(self):
        """Show status of all services"""
        console.print("\n📊 [bold]Service Status[/bold]")
        console.print("="*40)
        
        service_status = self.summary_panel.check_service_status()
        
        # Check router
        router_running = service_status["router"]
        console.print(f"Router (port 8000): {'🟢 Running' if router_running else '🔴 Stopped'}")
        
        # Check for llama-server processes
        llama_servers = service_status["llama_servers"]
        total_instances = service_status["total_instances"]
        console.print(f"Llama Servers: {'🟢' if llama_servers > 0 else '🔴'} {llama_servers}/{total_instances} running")
        
        # Show health
        health = self.summary_panel.get_system_health()
        console.print(f"System Health: {health['health_percentage']:.1f}% ({health['status']})")
    
    def monitor_dashboard(self):
        """Step 2: Monitor - open Rich-based TUI dashboard"""
        console.print("\n📊 [bold]Live Monitoring Dashboard[/bold]")
        console.print("="*60)
        
        service_status = self.summary_panel.check_service_status()
        if not service_status["router"] and service_status["llama_servers"] == 0:
            console.print("[red]❌ Services not running! Please start services first (option 1).[/red]")
            return
        
        console.print("[blue]Starting live dashboard... Press Ctrl+C to stop[/blue]")
        time.sleep(1)
        
        try:
            self.summary_panel.run_live_dashboard()
        except KeyboardInterrupt:
            console.print("\n[yellow]Live dashboard stopped[/yellow]")
    
    def tenant_manager_menu(self):
        """Step 3: Tenant Manager - runtime tenant management"""
        console.print("\n👥 [bold]Tenant Manager[/bold]")
        console.print("="*60)
        
        while True:
            console.print("\n[bold]Tenant Management:[/bold]")
            console.print("[bold green]1.[/bold green] List Tenants")
            console.print("[bold green]2.[/bold green] Add Tenant")
            console.print("[bold green]3.[/bold green] Remove Tenant")
            console.print("[bold green]4.[/bold green] Update Tenant")
            console.print("[bold green]5.[/bold green] Tenant Statistics")
            console.print("[bold red]0.[/bold red] Back to Main Menu")
            
            choice = console.input("\n[bold blue]Select an option[/bold blue]: ")
            
            if choice == "1":
                self.tenant_manager.list_tenants()
            elif choice == "2":
                self.add_tenant_interactive()
            elif choice == "3":
                self.remove_tenant_interactive()
            elif choice == "4":
                self.update_tenant_interactive()
            elif choice == "5":
                self.tenant_manager.status()
            elif choice == "0":
                break
            else:
                console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")
    
    def add_tenant_interactive(self):
        """Interactive tenant addition"""
        console.print("\n🔧 [bold]Add New Tenant[/bold]")
        console.print("="*40)
        
        alias = Prompt.ask("Tenant alias")
        quota = float(Prompt.ask("Quota percentage (0-100)", default="50"))
        priority = Prompt.ask("Priority (low/normal/high)", choices=["low", "normal", "high"], default="normal")
        
        success = self.tenant_manager.add_tenant(alias, quota, priority)
        if success:
            self.tenant_manager.save_tenants()
            # Update profile
            self.update_profile_with_tenants()
    
    def remove_tenant_interactive(self):
        """Interactive tenant removal"""
        console.print("\n🗑️ [bold]Remove Tenant[/bold]")
        console.print("="*40)
        
        # List tenants
        self.tenant_manager.list_tenants()
        
        alias = Prompt.ask("\nEnter tenant alias to remove")
        if Confirm.ask(f"Are you sure you want to remove tenant '{alias}'?"):
            success = self.tenant_manager.remove_tenant(alias)
            if success:
                self.tenant_manager.save_tenants()
                # Update profile
                self.update_profile_with_tenants()
    
    def update_tenant_interactive(self):
        """Interactive tenant update"""
        console.print("\n🔧 [bold]Update Tenant[/bold]")
        console.print("="*40)
        
        # List tenants
        self.tenant_manager.list_tenants()
        
        alias = Prompt.ask("\nEnter tenant alias to update")
        tenant = self.tenant_manager.get_tenant(alias)
        
        if not tenant:
            console.print(f"[red]❌ Tenant '{alias}' not found![/red]")
            return
        
        console.print(f"\n[bold]Updating tenant: {tenant.name}[/bold]")
        
        # Update fields
        quota = Prompt.ask("New quota percentage (0-100)", default=str(tenant.quota.quota_percentage if tenant.quota else 50))
        priority = Prompt.ask("New priority (low/normal/high)", choices=["low", "normal", "high"], default=tenant.policy.priority if tenant.policy else "normal")
        
        success = self.tenant_manager.update_tenant(alias, float(quota), priority)
        if success:
            self.tenant_manager.save_tenants()
            # Update profile
            self.update_profile_with_tenants()
    
    def update_profile_with_tenants(self):
        """Update profile with current tenant configuration"""
        if not self.current_profile:
            return
        
        # Get current profile
        profile_data = self.profile_manager.load_profile(self.current_profile)
        if not profile_data:
            return
        
        # Update tenant config
        tenant_summary = self.tenant_manager.get_system_summary()
        profile_data["tenant_config"] = {
            "tenant_count": tenant_summary["tenant_count"],
            "active_tenants": tenant_summary["active_tenants"],
            "tenants": tenant_summary["tenants"]
        }
        
        # Save updated profile
        self.profile_manager.save_profile(self.current_profile, profile_data)
    
    def advanced_options(self):
        """Step 4: Advanced Options"""
        console.print("\n🔧 [bold]Advanced Options[/bold]")
        console.print("="*60)
        
        while True:
            console.print("\n[bold]Advanced Options:[/bold]")
            console.print("[bold green]1.[/bold green] Profile Management")
            console.print("[bold green]2.[/bold green] System Diagnostics")
            console.print("[bold green]3.[/bold green] Export/Snapshot")
            console.print("[bold green]4.[/bold green] Import Profile")
            console.print("[bold green]5.[/bold green] Cleanup Old Snapshots")
            console.print("[bold red]0.[/bold red] Back to Main Menu")
            
            choice = console.input("\n[bold blue]Select an option[/bold blue]: ")
            
            if choice == "1":
                self.profile_management()
            elif choice == "2":
                self.system_diagnostics()
            elif choice == "3":
                self.export_snapshot()
            elif choice == "4":
                self.import_profile()
            elif choice == "5":
                self.cleanup_snapshots()
            elif choice == "0":
                break
            else:
                console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")
    
    def profile_management(self):
        """Profile management interface"""
        console.print("\n📁 [bold]Profile Management[/bold]")
        console.print("="*40)
        
        while True:
            console.print("\n[bold]Profile Options:[/bold]")
            console.print("[bold green]1.[/bold green] List Profiles")
            console.print("[bold green]2.[/bold green] Load Profile")
            console.print("[bold green]3.[/bold green] Delete Profile")
            console.print("[bold green]4.[/bold green] Create New Profile")
            console.print("[bold red]0.[/bold red] Back")
            
            choice = console.input("\n[bold blue]Select an option[/bold blue]: ")
            
            if choice == "1":
                self.profile_manager.display_profiles()
            elif choice == "2":
                self.load_profile_interactive()
            elif choice == "3":
                self.delete_profile_interactive()
            elif choice == "4":
                self.create_profile_interactive()
            elif choice == "0":
                break
            else:
                console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")
    
    def load_profile_interactive(self):
        """Interactive profile loading"""
        console.print("\n📂 [bold]Load Profile[/bold]")
        console.print("="*30)
        
        # List profiles
        self.profile_manager.display_profiles()
        
        profile_name = Prompt.ask("\nEnter profile name to load")
        profile_data = self.profile_manager.load_profile(profile_name)
        
        if profile_data:
            self.current_profile = profile_name
            console.print(f"[green]✅ Loaded profile: {profile_name}[/green]")
        else:
            console.print(f"[red]❌ Failed to load profile: {profile_name}[/red]")
    
    def delete_profile_interactive(self):
        """Interactive profile deletion"""
        console.print("\n🗑️ [bold]Delete Profile[/bold]")
        console.print("="*30)
        
        # List profiles
        self.profile_manager.display_profiles()
        
        profile_name = Prompt.ask("\nEnter profile name to delete")
        if Confirm.ask(f"Are you sure you want to delete profile '{profile_name}'?"):
            success = self.profile_manager.delete_profile(profile_name)
            if success and profile_name == self.current_profile:
                self.current_profile = None
                console.print("[yellow]⚠️ Current profile deleted. Please load another profile.[/yellow]")
    
    def create_profile_interactive(self):
        """Interactive profile creation"""
        console.print("\n🔧 [bold]Create New Profile[/bold]")
        console.print("="*40)
        
        profile_name = Prompt.ask("Profile name")
        
        # Use current system state
        gpu_summary = self.gpu_detector.get_system_summary()
        model_summary = self.model_detector.get_system_summary()
        tenant_summary = self.tenant_manager.get_system_summary()
        
        success = self.profile_manager.create_profile(
            profile_name,
            gpu_summary,
            model_summary,
            tenant_summary
        )
        
        if success:
            self.current_profile = profile_name
            console.print(f"[green]✅ Created profile: {profile_name}[/green]")
    
    def system_diagnostics(self):
        """System diagnostics"""
        console.print("\n🔍 [bold]System Diagnostics[/bold]")
        console.print("="*40)
        
        # GPU diagnostics
        console.print("\n[bold]GPU Status:[/bold]")
        self.gpu_detector.display_gpu_summary()
        
        # Model diagnostics
        console.print("\n[bold]Model Status:[/bold]")
        self.model_detector.display_models()
        
        # Tenant diagnostics
        console.print("\n[bold]Tenant Status:[/bold]")
        self.tenant_manager.list_tenants()
        
        # Service diagnostics
        console.print("\n[bold]Service Status:[/bold]")
        self.show_service_status()
        
        # System health
        health = self.summary_panel.get_system_health()
        console.print(f"\n[bold]System Health: {health['health_percentage']:.1f}% ({health['status']})[/bold]")
    
    def export_snapshot(self):
        """Export/Snapshot - save active profile + metrics"""
        console.print("\n💾 [bold]Export/Snapshot[/bold]")
        console.print("="*40)
        
        if not self.current_profile:
            console.print("[red]❌ No active profile! Please load a profile first.[/red]")
            return
        
        # Get current state
        gpu_summary = self.gpu_detector.get_system_summary()
        model_summary = self.model_detector.get_system_summary()
        tenant_summary = self.tenant_manager.get_system_summary()
        service_status = self.summary_panel.check_service_status()
        
        # Create metrics
        metrics = {
            "timestamp": time.time(),
            "profile": self.current_profile,
            "services_running": service_status["router"],
            "health": self.summary_panel.get_system_health()
        }
        
        # Create snapshot
        snapshot_path = self.profile_manager.create_snapshot(
            self.current_profile,
            gpu_summary,
            model_summary,
            tenant_summary,
            service_status,
            metrics
        )
        
        if snapshot_path:
            console.print(f"[green]✅ Snapshot created successfully![/green]")
            console.print(f"[blue]📁 Location: {snapshot_path}[/blue]")
    
    def import_profile(self):
        """Import profile from external location"""
        console.print("\n📥 [bold]Import Profile[/bold]")
        console.print("="*30)
        
        import_path = Prompt.ask("Path to profile file")
        profile_name = Prompt.ask("Profile name (or press Enter to use filename)", default="")
        
        success = self.profile_manager.import_profile(import_path, profile_name or None)
        if success:
            console.print("[green]✅ Profile imported successfully![/green]")
    
    def cleanup_snapshots(self):
        """Cleanup old snapshots"""
        console.print("\n🧹 [bold]Cleanup Old Snapshots[/bold]")
        console.print("="*40)
        
        keep_days = IntPrompt.ask("Keep snapshots newer than (days)", default=30)
        
        if Confirm.ask(f"Delete snapshots older than {keep_days} days?"):
            self.profile_manager.cleanup_old_snapshots(keep_days)

def main():
    """Main entry point"""
    wizard = MultiTenantWizard()
    wizard.run()

if __name__ == "__main__":
    main()
