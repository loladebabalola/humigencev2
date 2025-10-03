# inference/profile_manager.py

import os
import json
import time
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

@dataclass
class ProfileSnapshot:
    """Profile snapshot with metadata"""
    timestamp: str
    profile_name: str
    gpu_config: Dict[str, Any]
    model_config: Dict[str, Any]
    tenant_config: Dict[str, Any]
    service_status: Dict[str, Any]
    metrics: Dict[str, Any]
    exported_at: float

class ProfileManager:
    """Profile persistence system for tenant state and snapshots"""
    
    def __init__(self, profiles_dir: str = "profiles", runs_dir: str = "runs"):
        self.profiles_dir = Path(profiles_dir)
        self.runs_dir = Path(runs_dir)
        self.current_profile = None
        
        # Ensure directories exist
        self.profiles_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
    
    def create_profile(self, profile_name: str, gpu_config: Dict, model_config: Dict, tenant_config: Dict) -> bool:
        """Create a new profile configuration"""
        try:
            profile_data = {
                "profile_name": profile_name,
                "created_at": time.time(),
                "gpu_config": gpu_config,
                "model_config": model_config,
                "tenant_config": tenant_config,
                "router_config": {
                    "entrypoint": "localhost:8000",
                    "strategy": "tenant_sticky_then_latency"
                }
            }
            
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            with open(profile_path, 'w') as f:
                yaml.dump(profile_data, f, default_flow_style=False)
            
            self.current_profile = profile_name
            console.print(f"[green]✅ Created profile: {profile_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create profile: {e}[/red]")
            return False
    
    def load_profile(self, profile_name: str) -> Optional[Dict]:
        """Load a profile configuration"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if not profile_path.exists():
                console.print(f"[red]❌ Profile not found: {profile_name}[/red]")
                return None
            
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)
            
            self.current_profile = profile_name
            console.print(f"[green]✅ Loaded profile: {profile_name}[/green]")
            return profile_data
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load profile: {e}[/red]")
            return None
    
    def save_profile(self, profile_name: str, profile_data: Dict) -> bool:
        """Save profile configuration"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            with open(profile_path, 'w') as f:
                yaml.dump(profile_data, f, default_flow_style=False)
            
            console.print(f"[green]✅ Saved profile: {profile_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save profile: {e}[/red]")
            return False
    
    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        profiles = []
        for profile_file in self.profiles_dir.glob("*.yaml"):
            profiles.append(profile_file.stem)
        return sorted(profiles)
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if not profile_path.exists():
                console.print(f"[red]❌ Profile not found: {profile_name}[/red]")
                return False
            
            profile_path.unlink()
            console.print(f"[green]✅ Deleted profile: {profile_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to delete profile: {e}[/red]")
            return False
    
    def create_snapshot(self, profile_name: str, gpu_config: Dict, model_config: Dict, 
                       tenant_config: Dict, service_status: Dict, metrics: Dict) -> str:
        """Create a profile snapshot with current state"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"{profile_name}_snapshot_{timestamp}"
            snapshot_dir = self.runs_dir / snapshot_name
            snapshot_dir.mkdir(exist_ok=True)
            
            # Create snapshot data
            snapshot = ProfileSnapshot(
                timestamp=timestamp,
                profile_name=profile_name,
                gpu_config=gpu_config,
                model_config=model_config,
                tenant_config=tenant_config,
                service_status=service_status,
                metrics=metrics,
                exported_at=time.time()
            )
            
            # Save snapshot
            snapshot_path = snapshot_dir / "snapshot.json"
            with open(snapshot_path, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2)
            
            # Copy profile
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if profile_path.exists():
                shutil.copy2(profile_path, snapshot_dir / "profile.yaml")
            
            # Save metrics separately
            metrics_path = snapshot_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            console.print(f"[green]✅ Created snapshot: {snapshot_name}[/green]")
            console.print(f"[blue]📁 Snapshot directory: {snapshot_dir}[/blue]")
            
            return str(snapshot_dir)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create snapshot: {e}[/red]")
            return ""
    
    def load_snapshot(self, snapshot_dir: str) -> Optional[ProfileSnapshot]:
        """Load a profile snapshot"""
        try:
            snapshot_path = Path(snapshot_dir) / "snapshot.json"
            if not snapshot_path.exists():
                console.print(f"[red]❌ Snapshot not found: {snapshot_dir}[/red]")
                return None
            
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            snapshot = ProfileSnapshot(**snapshot_data)
            console.print(f"[green]✅ Loaded snapshot: {snapshot.timestamp}[/green]")
            return snapshot
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load snapshot: {e}[/red]")
            return None
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        snapshots = []
        for snapshot_dir in self.runs_dir.iterdir():
            if snapshot_dir.is_dir() and snapshot_dir.name.endswith("_snapshot_"):
                snapshot_path = snapshot_dir / "snapshot.json"
                if snapshot_path.exists():
                    try:
                        with open(snapshot_path, 'r') as f:
                            snapshot_data = json.load(f)
                        snapshots.append({
                            "name": snapshot_dir.name,
                            "path": str(snapshot_dir),
                            "timestamp": snapshot_data.get("timestamp", "unknown"),
                            "profile_name": snapshot_data.get("profile_name", "unknown"),
                            "exported_at": snapshot_data.get("exported_at", 0)
                        })
                    except:
                        continue
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x["exported_at"], reverse=True)
        return snapshots
    
    def export_profile(self, profile_name: str, export_path: str) -> bool:
        """Export profile to external location"""
        try:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if not profile_path.exists():
                console.print(f"[red]❌ Profile not found: {profile_name}[/red]")
                return False
            
            export_file = Path(export_path)
            shutil.copy2(profile_path, export_file)
            
            console.print(f"[green]✅ Exported profile to: {export_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export profile: {e}[/red]")
            return False
    
    def import_profile(self, import_path: str, profile_name: Optional[str] = None) -> bool:
        """Import profile from external location"""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                console.print(f"[red]❌ Import file not found: {import_path}[/red]")
                return False
            
            if not profile_name:
                profile_name = import_file.stem
            
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            shutil.copy2(import_file, profile_path)
            
            console.print(f"[green]✅ Imported profile: {profile_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to import profile: {e}[/red]")
            return False
    
    def display_profiles(self):
        """Display all profiles in a table"""
        profiles = self.list_profiles()
        
        if not profiles:
            console.print("[yellow]⚠️ No profiles found[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Available Profiles")
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Created", style="white", width=20)
        table.add_column("Size", style="green", width=10)
        table.add_column("Status", style="yellow", width=10)
        
        for profile_name in profiles:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if profile_path.exists():
                stat = profile_path.stat()
                created = time.ctime(stat.st_ctime)
                size = f"{stat.st_size / 1024:.1f}KB"
                status = "⭐ Current" if profile_name == self.current_profile else "📁"
                
                table.add_row(profile_name, created, size, status)
        
        console.print(table)
    
    def display_snapshots(self):
        """Display all snapshots in a table"""
        snapshots = self.list_snapshots()
        
        if not snapshots:
            console.print("[yellow]⚠️ No snapshots found[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Available Snapshots")
        table.add_column("Name", style="cyan", width=25)
        table.add_column("Profile", style="white", width=15)
        table.add_column("Timestamp", style="green", width=20)
        table.add_column("Size", style="blue", width=10)
        
        for snapshot in snapshots:
            snapshot_path = Path(snapshot["path"])
            size = sum(f.stat().st_size for f in snapshot_path.rglob('*') if f.is_file())
            size_str = f"{size / 1024:.1f}KB"
            
            table.add_row(
                snapshot["name"],
                snapshot["profile_name"],
                snapshot["timestamp"],
                size_str
            )
        
        console.print(table)
    
    def get_profile_summary(self, profile_name: str) -> Dict[str, Any]:
        """Get a summary of a profile"""
        profile = self.load_profile(profile_name)
        if not profile:
            return {}
        
        return {
            "name": profile_name,
            "created_at": profile.get("created_at", 0),
            "gpu_count": len(profile.get("gpu_config", {}).get("gpus", [])),
            "model_name": profile.get("model_config", {}).get("name", "N/A"),
            "tenant_count": len(profile.get("tenant_config", {}).get("tenants", {})),
            "router_strategy": profile.get("router_config", {}).get("strategy", "N/A")
        }
    
    def cleanup_old_snapshots(self, keep_days: int = 30):
        """Clean up old snapshots"""
        try:
            cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
            snapshots = self.list_snapshots()
            
            deleted_count = 0
            for snapshot in snapshots:
                if snapshot["exported_at"] < cutoff_time:
                    snapshot_path = Path(snapshot["path"])
                    if snapshot_path.exists():
                        shutil.rmtree(snapshot_path)
                        deleted_count += 1
            
            if deleted_count > 0:
                console.print(f"[green]✅ Cleaned up {deleted_count} old snapshots[/green]")
            else:
                console.print("[blue]ℹ️ No old snapshots to clean up[/blue]")
                
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup snapshots: {e}[/red]")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a complete system summary"""
        profiles = self.list_profiles()
        snapshots = self.list_snapshots()
        
        return {
            "profile_count": len(profiles),
            "snapshot_count": len(snapshots),
            "current_profile": self.current_profile,
            "profiles": profiles,
            "recent_snapshots": snapshots[:5]  # Last 5 snapshots
        }

def main():
    """Test profile manager"""
    manager = ProfileManager()
    
    console.print("[bold cyan]Profile Manager Test[/bold cyan]")
    console.print("=" * 50)
    
    # Test profile creation
    test_profile = {
        "gpu_config": {"gpus": [{"index": 0, "name": "RTX 5090", "port": 8000}]},
        "model_config": {"name": "test-model", "path": "/models/test.gguf"},
        "tenant_config": {"tenants": {"default": {"name": "Default"}}}
    }
    
    manager.create_profile("test_profile", **test_profile)
    
    # Display profiles
    manager.display_profiles()
    
    # Test snapshot creation
    manager.create_snapshot(
        "test_profile",
        test_profile["gpu_config"],
        test_profile["model_config"],
        test_profile["tenant_config"],
        {"router": True, "llama_servers": 1},
        {"total_requests": 100, "avg_latency": 0.5}
    )
    
    # Display snapshots
    manager.display_snapshots()
    
    # Show system summary
    summary = manager.get_system_summary()
    console.print(f"\n[blue]System Summary:[/blue]")
    console.print(f"  Profiles: {summary['profile_count']}")
    console.print(f"  Snapshots: {summary['snapshot_count']}")
    console.print(f"  Current: {summary['current_profile']}")

if __name__ == "__main__":
    main()
