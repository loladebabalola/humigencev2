# inference/tenants.py

import os
import yaml
import time
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
import inquirer

console = Console()

@dataclass
class TenantQuota:
    """Tenant quota configuration"""
    rps: int = 10  # Requests per second
    max_context: int = 131072  # Maximum context length
    max_tokens_per_request: int = 2048  # Maximum tokens per request
    daily_limit: int = 10000  # Daily request limit

@dataclass
class TenantPolicy:
    """Tenant policy configuration"""
    priority: int = 1  # Priority level (1=highest, 5=lowest)
    allow_streaming: bool = True  # Allow streaming responses
    max_concurrent_requests: int = 5  # Maximum concurrent requests
    timeout_seconds: int = 30  # Request timeout

@dataclass
class Tenant:
    """Tenant configuration"""
    tenant_id: str
    name: str
    api_key: Optional[str] = None
    quota: Optional[TenantQuota] = None
    policy: Optional[TenantPolicy] = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    is_active: bool = True

class TenantManager:
    """Manager for multi-tenant configuration and policies"""
    
    def __init__(self, tenants_file: str = "tenants.yaml"):
        self.tenants_file = Path(tenants_file)
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load existing tenants
        self.load_tenants()
    
    def load_tenants(self):
        """Load tenants from YAML file"""
        if not self.tenants_file.exists():
            console.print(f"[yellow]⚠️ Tenants file not found: {self.tenants_file}[/yellow]")
            self.create_default_tenants()
            return
        
        try:
            with open(self.tenants_file, 'r') as f:
                data = yaml.safe_load(f)
            
            tenants_data = data.get('tenants', {})
            
            for tenant_id, tenant_data in tenants_data.items():
                # Convert dict to Tenant object
                quota_data = tenant_data.get('quota', {})
                policy_data = tenant_data.get('policy', {})
                
                quota = TenantQuota(**quota_data) if quota_data else TenantQuota()
                policy = TenantPolicy(**policy_data) if policy_data else TenantPolicy()
                
                tenant = Tenant(
                    tenant_id=tenant_id,
                    name=tenant_data.get('name', tenant_id),
                    api_key=tenant_data.get('api_key'),
                    quota=quota,
                    policy=policy,
                    created_at=tenant_data.get('created_at', time.time()),
                    last_accessed=tenant_data.get('last_accessed', 0.0),
                    is_active=tenant_data.get('is_active', True)
                )
                
                self.tenants[tenant_id] = tenant
                self.tenant_stats[tenant_id] = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'current_rps': 0.0,
                    'last_request_time': 0.0,
                    'error_count': 0
                }
            
            console.print(f"[green]✅ Loaded {len(self.tenants)} tenant(s)[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load tenants: {e}[/red]")
            self.create_default_tenants()
    
    def save_tenants(self):
        """Save tenants to YAML file"""
        try:
            data = {
                'tenants': {},
                'metadata': {
                    'version': '1.0.0',
                    'last_updated': time.time(),
                    'total_tenants': len(self.tenants)
                }
            }
            
            for tenant_id, tenant in self.tenants.items():
                tenant_data = {
                    'name': tenant.name,
                    'api_key': tenant.api_key,
                    'quota': asdict(tenant.quota) if tenant.quota else {},
                    'policy': asdict(tenant.policy) if tenant.policy else {},
                    'created_at': tenant.created_at,
                    'last_accessed': tenant.last_accessed,
                    'is_active': tenant.is_active
                }
                data['tenants'][tenant_id] = tenant_data
            
            with open(self.tenants_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            console.print(f"[green]✅ Saved {len(self.tenants)} tenant(s) to {self.tenants_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save tenants: {e}[/red]")
    
    def create_default_tenants(self):
        """Create default tenant configuration"""
        console.print("[blue]🔧 Creating default tenant configuration...[/blue]")
        
        # Create default tenant
        default_tenant = Tenant(
            tenant_id="default",
            name="Default Tenant",
            api_key=None,  # No API key required for default
            quota=TenantQuota(rps=10, max_context=131072, max_tokens_per_request=2048, daily_limit=10000),
            policy=TenantPolicy(priority=1, allow_streaming=True, max_concurrent_requests=5, timeout_seconds=30),
            created_at=time.time(),
            is_active=True
        )
        
        self.tenants["default"] = default_tenant
        self.tenant_stats["default"] = {
            'total_requests': 0,
            'total_tokens': 0,
            'current_rps': 0.0,
            'last_request_time': 0.0,
            'error_count': 0
        }
        
        # Save default configuration
        self.save_tenants()
    
    def create_tenant(self, tenant_id: str, name: str, **kwargs) -> Tenant:
        """Create a new tenant"""
        if tenant_id in self.tenants:
            console.print(f"[red]❌ Tenant '{tenant_id}' already exists![/red]")
            return None
        
        # Generate API key if not provided
        api_key = kwargs.get('api_key')
        if not api_key:
            api_key = self.generate_api_key()
        
        # Create tenant with default values
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_key=api_key,
            quota=TenantQuota(**kwargs.get('quota', {})),
            policy=TenantPolicy(**kwargs.get('policy', {})),
            created_at=time.time(),
            is_active=True
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_stats[tenant_id] = {
            'total_requests': 0,
            'total_tokens': 0,
            'current_rps': 0.0,
            'last_request_time': 0.0,
            'error_count': 0
        }
        
        console.print(f"[green]✅ Created tenant '{tenant_id}'[/green]")
        return tenant
    
    def update_tenant(self, tenant_id: str, **kwargs) -> bool:
        """Update an existing tenant"""
        if tenant_id not in self.tenants:
            console.print(f"[red]❌ Tenant '{tenant_id}' not found![/red]")
            return False
        
        tenant = self.tenants[tenant_id]
        
        # Update fields
        if 'name' in kwargs:
            tenant.name = kwargs['name']
        if 'api_key' in kwargs:
            tenant.api_key = kwargs['api_key']
        if 'quota' in kwargs:
            quota_data = kwargs['quota']
            if tenant.quota:
                for key, value in quota_data.items():
                    setattr(tenant.quota, key, value)
            else:
                tenant.quota = TenantQuota(**quota_data)
        if 'policy' in kwargs:
            policy_data = kwargs['policy']
            if tenant.policy:
                for key, value in policy_data.items():
                    setattr(tenant.policy, key, value)
            else:
                tenant.policy = TenantPolicy(**policy_data)
        if 'is_active' in kwargs:
            tenant.is_active = kwargs['is_active']
        
        console.print(f"[green]✅ Updated tenant '{tenant_id}'[/green]")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant"""
        if tenant_id not in self.tenants:
            console.print(f"[red]❌ Tenant '{tenant_id}' not found![/red]")
            return False
        
        if tenant_id == "default":
            console.print(f"[red]❌ Cannot delete default tenant![/red]")
            return False
        
        del self.tenants[tenant_id]
        if tenant_id in self.tenant_stats:
            del self.tenant_stats[tenant_id]
        
        console.print(f"[green]✅ Deleted tenant '{tenant_id}'[/green]")
        return True
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID"""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get a tenant by API key"""
        for tenant in self.tenants.values():
            if tenant.api_key == api_key:
                return tenant
        return None
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants"""
        return list(self.tenants.values())
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def validate_tenant_request(self, tenant_id: str, request_size: int) -> tuple[bool, str]:
        """Validate if a tenant can make a request"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False, "Tenant not found"
        
        if not tenant.is_active:
            return False, "Tenant is inactive"
        
        if not tenant.quota:
            return True, "No quota restrictions"
        
        # Check RPS limit
        stats = self.tenant_stats.get(tenant_id, {})
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
    
    def record_request(self, tenant_id: str, tokens: int, success: bool = True):
        """Record a request for quota tracking"""
        if tenant_id not in self.tenant_stats:
            return
        
        stats = self.tenant_stats[tenant_id]
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
    
    def display_tenants(self):
        """Display all tenants in a table"""
        if not self.tenants:
            console.print("[yellow]⚠️ No tenants configured[/yellow]")
            return
        
        table = Table(show_header=True, box=None)
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Name", style="white", width=20)
        table.add_column("API Key", style="green", width=20)
        table.add_column("RPS", style="blue", width=6)
        table.add_column("Max Context", style="yellow", width=12)
        table.add_column("Priority", style="red", width=8)
        table.add_column("Status", style="white", width=8)
        
        for tenant in self.tenants.values():
            api_key_display = tenant.api_key[:8] + "..." if tenant.api_key else "None"
            rps = tenant.quota.rps if tenant.quota else "N/A"
            max_context = f"{tenant.quota.max_context:,}" if tenant.quota else "N/A"
            priority = tenant.policy.priority if tenant.policy else "N/A"
            status = "🟢 Active" if tenant.is_active else "🔴 Inactive"
            
            table.add_row(
                tenant.tenant_id,
                tenant.name,
                api_key_display,
                str(rps),
                max_context,
                str(priority),
                status
            )
        
        console.print(table)
    
    def display_tenant_stats(self):
        """Display tenant statistics"""
        if not self.tenant_stats:
            console.print("[yellow]⚠️ No tenant statistics available[/yellow]")
            return
        
        table = Table(show_header=True, box=None)
        table.add_column("Tenant ID", style="cyan", width=12)
        table.add_column("Total Requests", style="blue", width=15)
        table.add_column("Total Tokens", style="green", width=15)
        table.add_column("Current RPS", style="yellow", width=12)
        table.add_column("Error Count", style="red", width=12)
        table.add_column("Last Request", style="white", width=20)
        
        for tenant_id, stats in self.tenant_stats.items():
            last_request = time.ctime(stats['last_request_time']) if stats['last_request_time'] > 0 else "Never"
            
            table.add_row(
                tenant_id,
                f"{stats['total_requests']:,}",
                f"{stats['total_tokens']:,}",
                f"{stats['current_rps']:.2f}",
                str(stats['error_count']),
                last_request
            )
        
        console.print(table)

def interactive_tenant_manager():
    """Interactive tenant management interface"""
    manager = TenantManager()
    
    while True:
        console.print("\n" + "="*60)
        console.print("[bold cyan]Tenant Management Console[/bold cyan]")
        console.print("="*60)
        
        console.print("\n[bold]Available Actions:[/bold]")
        console.print("[bold green]1.[/bold green] List Tenants")
        console.print("[bold green]2.[/bold green] Create Tenant")
        console.print("[bold green]3.[/bold green] Update Tenant")
        console.print("[bold green]4.[/bold green] Delete Tenant")
        console.print("[bold green]5.[/bold green] View Statistics")
        console.print("[bold green]6.[/bold green] Save Configuration")
        console.print("[bold red]0.[/bold red] Exit")
        
        choice = console.input("\n[bold blue]Select an option[/bold blue]: ")
        
        if choice == "1":
            manager.display_tenants()
        elif choice == "2":
            create_tenant_interactive(manager)
        elif choice == "3":
            update_tenant_interactive(manager)
        elif choice == "4":
            delete_tenant_interactive(manager)
        elif choice == "5":
            manager.display_tenant_stats()
        elif choice == "6":
            manager.save_tenants()
        elif choice == "0":
            console.print("[bold red]Exiting Tenant Manager...[/bold red]")
            break
        else:
            console.print("[yellow]⚠️ Invalid option. Please try again.[/yellow]")

def create_tenant_interactive(manager: TenantManager):
    """Interactive tenant creation"""
    console.print("\n🔧 [bold]Create New Tenant[/bold]")
    console.print("="*40)
    
    tenant_id = Prompt.ask("Tenant ID")
    name = Prompt.ask("Tenant Name", default=tenant_id)
    
    # Quota configuration
    console.print("\n[bold]Quota Configuration:[/bold]")
    rps = IntPrompt.ask("Requests per second", default=10)
    max_context = IntPrompt.ask("Max context length", default=131072)
    max_tokens = IntPrompt.ask("Max tokens per request", default=2048)
    daily_limit = IntPrompt.ask("Daily request limit", default=10000)
    
    # Policy configuration
    console.print("\n[bold]Policy Configuration:[/bold]")
    priority = IntPrompt.ask("Priority (1=highest, 5=lowest)", default=1)
    allow_streaming = Confirm.ask("Allow streaming", default=True)
    max_concurrent = IntPrompt.ask("Max concurrent requests", default=5)
    timeout = IntPrompt.ask("Request timeout (seconds)", default=30)
    
    # Generate API key
    generate_key = Confirm.ask("Generate API key", default=True)
    api_key = manager.generate_api_key() if generate_key else None
    
    # Create tenant
    tenant = manager.create_tenant(
        tenant_id=tenant_id,
        name=name,
        api_key=api_key,
        quota={
            'rps': rps,
            'max_context': max_context,
            'max_tokens_per_request': max_tokens,
            'daily_limit': daily_limit
        },
        policy={
            'priority': priority,
            'allow_streaming': allow_streaming,
            'max_concurrent_requests': max_concurrent,
            'timeout_seconds': timeout
        }
    )
    
    if tenant:
        console.print(f"\n[green]✅ Tenant created successfully![/green]")
        if api_key:
            console.print(f"[blue]🔑 API Key: {api_key}[/blue]")
            console.print("[yellow]⚠️ Save this API key - it won't be shown again![/yellow]")

def update_tenant_interactive(manager: TenantManager):
    """Interactive tenant update"""
    console.print("\n🔧 [bold]Update Tenant[/bold]")
    console.print("="*40)
    
    # List tenants
    manager.display_tenants()
    
    tenant_id = Prompt.ask("\nEnter tenant ID to update")
    tenant = manager.get_tenant(tenant_id)
    
    if not tenant:
        console.print(f"[red]❌ Tenant '{tenant_id}' not found![/red]")
        return
    
    console.print(f"\n[bold]Updating tenant: {tenant.name}[/bold]")
    
    # Update fields
    new_name = Prompt.ask("Name", default=tenant.name)
    
    # Quota updates
    console.print("\n[bold]Quota Configuration:[/bold]")
    rps = IntPrompt.ask("Requests per second", default=tenant.quota.rps if tenant.quota else 10)
    max_context = IntPrompt.ask("Max context length", default=tenant.quota.max_context if tenant.quota else 131072)
    
    # Policy updates
    console.print("\n[bold]Policy Configuration:[/bold]")
    priority = IntPrompt.ask("Priority", default=tenant.policy.priority if tenant.policy else 1)
    is_active = Confirm.ask("Active", default=tenant.is_active)
    
    # Update tenant
    success = manager.update_tenant(
        tenant_id=tenant_id,
        name=new_name,
        quota={'rps': rps, 'max_context': max_context},
        policy={'priority': priority},
        is_active=is_active
    )
    
    if success:
        console.print(f"\n[green]✅ Tenant updated successfully![/green]")

def delete_tenant_interactive(manager: TenantManager):
    """Interactive tenant deletion"""
    console.print("\n🗑️ [bold]Delete Tenant[/bold]")
    console.print("="*40)
    
    # List tenants
    manager.display_tenants()
    
    tenant_id = Prompt.ask("\nEnter tenant ID to delete")
    
    if tenant_id == "default":
        console.print("[red]❌ Cannot delete default tenant![/red]")
        return
    
    if Confirm.ask(f"Are you sure you want to delete tenant '{tenant_id}'?"):
        success = manager.delete_tenant(tenant_id)
        if success:
            console.print(f"\n[green]✅ Tenant deleted successfully![/green]")

if __name__ == "__main__":
    interactive_tenant_manager()
