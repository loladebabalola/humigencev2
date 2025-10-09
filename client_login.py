#!/usr/bin/env python3
"""
Humigence Client Login System
=============================

This script provides a streamlined SSH-based authentication system
where clients can directly access the AI model through device credentials.

Usage:
    ssh user@server
    humigence-client
    # Enter device name and API key
    # Get direct access to AI model
"""

import os
import sys
import json
import time
import getpass
import requests
import termios
import tty
from pathlib import Path
from typing import Dict, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager

console = Console()

def get_password_input(prompt: str) -> str:
    """Get password input with hidden characters (works in all environments)"""
    try:
        # Try getpass first (works in most interactive terminals)
        # Note: getpass may not support paste in all terminals
        return getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt):
        # Fallback for non-interactive environments
        console.print(f"[yellow]⚠️ Using fallback password input (characters will be visible, paste supported)[/yellow]")
        return input(f"{prompt} (visible): ")
    except Exception:
        # Ultimate fallback
        console.print(f"[yellow]⚠️ Using fallback password input (characters will be visible, paste supported)[/yellow]")
        return input(f"{prompt} (visible): ")

def get_password_input_with_paste_option(prompt: str) -> str:
    """Get password input with option to use paste-friendly method"""
    console.print(f"\n[bold cyan]Password Input Options:[/bold cyan]")
    console.print("[blue]1.[/blue] Hidden input (secure, may not support paste)")
    console.print("[blue]2.[/blue] Visible input (less secure, supports paste)")
    
    choice = input("Choose input method (1 or 2, default=1): ").strip()
    
    if choice == "2":
        console.print(f"[yellow]⚠️ Using visible input (paste supported)[/yellow]")
        return input(f"{prompt} (visible): ")
    else:
        return get_password_input(prompt)

class HumigenceClientLogin:
    """SSH-based client authentication and AI access system"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.server_url = self.device_manager.base_endpoint
        self.current_device = None
        self.session_active = False
        self.session_file = Path("client_session.json")
        self.load_session()
    
    def load_session(self):
        """Load persistent session if available"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                
                device_id = session_data.get('device_id')
                api_key = session_data.get('api_key')
                
                if device_id and api_key:
                    # Verify the session is still valid
                    device = self.device_manager.get_device_by_api_key(api_key)
                    if device and device.device_id == device_id and device.is_active:
                        self.current_device = device
                        self.session_active = True
                        console.print(f"[green]✅ Restored session for {device.device_name}[/green]")
                        return True
                    else:
                        # Session is invalid, remove it
                        self.clear_session()
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load session: {e}[/yellow]")
        
        return False
    
    def save_session(self):
        """Save current session for persistence"""
        try:
            if self.current_device and self.session_active:
                session_data = {
                    'device_id': self.current_device.device_id,
                    'api_key': self.current_device.api_key,
                    'device_name': self.current_device.device_name,
                    'timestamp': time.time()
                }
                
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                console.print(f"[blue]💾 Session saved for {self.current_device.device_name}[/blue]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save session: {e}[/yellow]")
    
    def clear_session(self):
        """Clear the current session"""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            self.current_device = None
            self.session_active = False
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not clear session: {e}[/yellow]")
    
    def show_welcome(self):
        """Display welcome screen"""
        welcome = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Humigence AI Client Access                         ║
║                                                                              ║
║  🤖 Welcome to the Humigence AI Inference Server                            ║
║  📱 Please log in with your device credentials                              ║
║  🔐 Your device name and API key are provided by your administrator         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(welcome, style="bold blue"))
    
    def authenticate_device(self) -> bool:
        """Authenticate device with name and API key"""
        console.print("\n[bold cyan]🔐 Device Authentication[/bold cyan]")
        console.print("=" * 50)
        console.print("[yellow]💡 This is a one-time setup. After this, you'll be logged in permanently![/yellow]")
        console.print("")
        
        # Get device credentials
        device_name = Prompt.ask("Device Name")
        
        # Ask if user wants to paste the API key
        console.print(f"\n[bold cyan]API Key Input:[/bold cyan]")
        console.print("[blue]1.[/blue] Type manually (hidden characters)")
        console.print("[blue]2.[/blue] Paste from clipboard (visible characters)")
        
        paste_choice = input("Choose input method (1 or 2, default=1): ").strip()
        
        if paste_choice == "2":
            console.print(f"[yellow]⚠️ Using visible input for pasting (characters will be visible)[/yellow]")
            api_key = input("API Key (Password) - paste here: ")
        else:
            api_key = get_password_input("API Key (Password)")
        
        
        # Find device by API key
        device = self.device_manager.get_device_by_api_key(api_key)
        
        if not device:
            console.print("[red]❌ Invalid credentials. Please check your device name and API key.[/red]")
            return False
        
        # Verify device name matches (case-insensitive)
        if device.device_name.lower() != device_name.lower():
            console.print("[red]❌ Device name does not match. Please check your credentials.[/red]")
            return False
        
        # Check if device is active
        if not device.is_active:
            console.print("[red]❌ Device is inactive. Please contact your administrator.[/red]")
            return False
        
        # Authentication successful
        self.current_device = device
        self.session_active = True
        
        console.print(f"[green]✅ Authentication successful![/green]")
        console.print(f"[blue]Welcome, {device.device_name}![/blue]")
        console.print(f"[blue]Device ID: {device.device_id}[/blue]")
        console.print(f"[blue]Quota: {device.quota_percentage}%[/blue]")
        console.print(f"[blue]Priority: {device.priority}[/blue]")
        console.print(f"[green]🎉 You're now logged in permanently! No more credentials needed![/green]")
        
        # Update last accessed time
        self.device_manager.update_device_status(device.device_id, True)
        self.device_manager.save_devices()
        
        # Save session for persistence
        self.save_session()
        
        return True
    
    def show_main_menu(self):
        """Display main menu after authentication"""
        while self.session_active:
            console.print("\n[bold cyan]🤖 Humigence AI Client[/bold cyan]")
            console.print("=" * 50)
            console.print(f"[blue]Device: {self.current_device.device_name}[/blue]")
            console.print(f"[blue]Server: {self.server_url}[/blue]")
            console.print(f"[green]Session: Active (persistent until logout)[/green]")
            console.print("\n[bold]Options:[/bold]")
            console.print("[bold green]1.[/bold green] Chat with AI")
            console.print("[bold green]2.[/bold green] Quick Questions")
            console.print("[bold green]3.[/bold green] Check Server Status")
            console.print("[bold green]4.[/bold green] Device Information")
            console.print("[bold red]5.[/bold red] Logout")
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"])
            
            if choice == "1":
                self.chat_with_ai()
            elif choice == "2":
                self.quick_questions()
            elif choice == "3":
                self.check_server_status()
            elif choice == "4":
                self.show_device_info()
            elif choice == "5":
                self.logout()
    
    def chat_with_ai(self):
        """Interactive chat with AI"""
        console.print("\n[bold green]💬 Chat with AI[/bold green]")
        console.print("=" * 50)
        console.print("[yellow]Type 'quit' to return to main menu[/yellow]\n")
        
        while True:
            try:
                message = Prompt.ask("You")
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                
                if message.strip():
                    response = self.send_ai_request(message)
                    if response:
                        console.print(f"[bold blue]AI:[/bold blue] {response}\n")
                    else:
                        console.print("[red]❌ Failed to get response from AI[/red]\n")
            except KeyboardInterrupt:
                break
    
    def quick_questions(self):
        """Quick question templates"""
        console.print("\n[bold green]⚡ Quick Questions[/bold green]")
        console.print("=" * 50)
        
        questions = [
            "What's the weather like?",
            "Explain quantum computing",
            "Write a Python function to sort a list",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Tell me a joke",
            "What's the capital of France?",
            "Explain blockchain technology"
        ]
        
        # Show numbered list
        for i, question in enumerate(questions, 1):
            console.print(f"[blue]{i}.[/blue] {question}")
        
        console.print("[blue]0.[/blue] Custom question")
        console.print("[blue]q.[/blue] Back to main menu")
        
        choice = Prompt.ask("Select question")
        
        if choice.lower() == 'q':
            return
        elif choice == '0':
            custom_question = Prompt.ask("Enter your question")
            if custom_question.strip():
                response = self.send_ai_request(custom_question)
                if response:
                    console.print(f"\n[bold blue]AI:[/bold blue] {response}\n")
        elif choice.isdigit() and 1 <= int(choice) <= len(questions):
            question = questions[int(choice) - 1]
            console.print(f"\n[blue]Question:[/blue] {question}")
            response = self.send_ai_request(question)
            if response:
                console.print(f"[bold blue]AI:[/bold blue] {response}\n")
    
    def check_server_status(self):
        """Check server and device status"""
        console.print("\n[bold green]📊 Server Status[/bold green]")
        console.print("=" * 50)
        
        try:
            # Check server health
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Server is online[/green]")
                health_data = response.json()
                console.print(f"[blue]Status: {health_data.get('status', 'unknown')}[/blue]")
            else:
                console.print(f"[yellow]⚠️ Server returned status {response.status_code}[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Server is offline: {e}[/red]")
        
        # Show device info
        console.print(f"\n[blue]Device: {self.current_device.device_name}[/blue]")
        console.print(f"[blue]Device ID: {self.current_device.device_id}[/blue]")
        console.print(f"[blue]Quota: {self.current_device.quota_percentage}%[/blue]")
        console.print(f"[blue]Priority: {self.current_device.priority}[/blue]")
        console.print(f"[blue]Endpoint: {self.current_device.endpoint}[/blue]")
    
    def show_device_info(self):
        """Show detailed device information"""
        console.print("\n[bold green]📱 Device Information[/bold green]")
        console.print("=" * 50)
        
        device = self.current_device
        device_info = device.device_info
        
        table = Table(show_header=True, box=None)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="white", width=30)
        
        table.add_row("Device Name", device.device_name)
        table.add_row("Device ID", device.device_id)
        table.add_row("Device Type", device_info.device_type)
        table.add_row("Operating System", f"{device_info.os_name} {device_info.os_version}")
        table.add_row("Quota Percentage", f"{device.quota_percentage}%")
        table.add_row("Priority", device.priority)
        table.add_row("Status", "🟢 Active" if device.is_active else "🔴 Inactive")
        table.add_row("Endpoint", device.endpoint)
        table.add_row("Created", time.ctime(device.created_at))
        table.add_row("Last Accessed", time.ctime(device.last_accessed) if device.last_accessed > 0 else "Never")
        
        console.print(table)
    
    def clean_response(self, content: str) -> str:
        """Clean up the response by removing internal reasoning tokens"""
        import re
        
        # Pattern 1: Complete pattern with final response
        # <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>actual_response
        pattern1 = r'<\|channel\|>analysis<\|message\|>.*?<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)$'
        match1 = re.search(pattern1, content, re.DOTALL)
        
        if match1:
            return match1.group(1).strip()
        
        # Pattern 2: Just final part
        # <|channel|>final<|message|>actual_response
        pattern2 = r'<\|channel\|>final<\|message\|>(.*?)$'
        match2 = re.search(pattern2, content, re.DOTALL)
        
        if match2:
            return match2.group(1).strip()
        
        # Pattern 3: Analysis only (incomplete response) - extract quoted content
        # Look for quoted text within the analysis that looks like the actual answer
        if '<|channel|>analysis<|message|>' in content:
            # Look for quoted text that appears to be the actual answer
            quoted_pattern = r'"[^"]*"'
            quoted_matches = re.findall(quoted_pattern, content)
            
            if quoted_matches:
                # Return the last (most likely final) quoted text
                return quoted_matches[-1].strip('"')
            
            # If no quotes, look for text after "So I'd give a definition:" or similar
            definition_pattern = r'(?:So I\'d give a definition:|definition:|answer:)\s*["\']?([^"\']*)["\']?'
            def_match = re.search(definition_pattern, content, re.IGNORECASE)
            
            if def_match:
                return def_match.group(1).strip()
        
        # Pattern 4: Remove any remaining special tokens
        content = re.sub(r'<\|[^|]*\|>', '', content)
        
        # Pattern 5: Remove system prompt repetition
        content = re.sub(r'You are a helpful AI assistant\. Respond directly and concisely\. Do not show your internal reasoning process or use special tokens.*?Just provide clear, direct answers\.', '', content, flags=re.IGNORECASE)
        
        # Pattern 6: Remove common unhelpful responses
        unhelpful_patterns = [
            r"I'm sorry, but I can't provide that\.",
            r"I can't help with that\.",
            r"I don't have access to that information\.",
            r"I'm not able to provide that\."
        ]
        
        for pattern in unhelpful_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # If content is too short or unhelpful, return a default response
        if len(content) < 10 or content.lower() in ['ok', 'yes', 'no', 'i see', 'understood']:
            return "I'd be happy to help with that. Could you please provide more details about what you'd like to know?"
        
        return content
    
    def send_ai_request(self, message: str, max_tokens: int = 500) -> Optional[str]:
        """Send request to AI and return response"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.current_device.api_key}",
                "X-Tenant-ID": self.current_device.device_id
            }
            
            payload = {
                "model": "gpt-oss-20b-F16",
                "messages": [{"role": "user", "content": message}],
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            # Use the main endpoint instead of device-specific endpoint
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0].get('message', {}).get('content', '')
                    # Clean up the response by removing internal reasoning tokens
                    return self.clean_response(content)
            
            console.print(f"[red]❌ Request failed with status {response.status_code}[/red]")
            return None
            
        except Exception as e:
            console.print(f"[red]❌ Request failed: {e}[/red]")
            return None
    
    def logout(self):
        """Logout and end session"""
        if self.current_device:
            console.print(f"\n[blue]👋 Goodbye, {self.current_device.device_name}![/blue]")
            # Update last accessed time (but keep device active)
            self.current_device.last_accessed = time.time()
            self.device_manager.save_devices()
        
        # Clear the persistent session
        self.clear_session()
        self.session_active = False
        console.print("[green]✅ Logged out successfully[/green]")
        console.print("[yellow]💡 Session cleared - you'll need to log in again next time[/yellow]")
    
    def run(self):
        """Main entry point"""
        try:
            # Check if we already have a valid session
            if self.session_active and self.current_device:
                console.print(f"[green]✅ Welcome back, {self.current_device.device_name}![/green]")
                console.print(f"[blue]Server: {self.server_url}[/blue]")
                console.print(f"[green]Session: Active (persistent until logout)[/green]")
                self.show_main_menu()
                return
            
            # Only show welcome and authenticate if no valid session
            self.show_welcome()
            
            # Authenticate device
            if not self.authenticate_device():
                console.print("[red]❌ Authentication failed. Exiting.[/red]")
                return
            
            # Show main menu
            self.show_main_menu()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
            self.logout()
        except Exception as e:
            console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
            self.logout()

def main():
    """Main entry point"""
    client = HumigenceClientLogin()
    client.run()

if __name__ == "__main__":
    main()
