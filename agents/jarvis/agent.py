#!/usr/bin/env python3
"""
JARVIS - AI Agent
An intelligent AI agent
"""

import json
import sys
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

class JARVISAgent:
    """AI Agent: JARVIS"""
    
    def __init__(self, config_path: str = None):
        """Initialize the agent with configuration."""
        if config_path is None:
            # Find config.json relative to this script's location
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.json"
        
        self.config = self._load_config(str(config_path))
        self.capabilities = self.config.get("capabilities", [])
        self.model = self.config.get("model", "GPT-4")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
    
    def process_request(self, request: str) -> str:
        """Process a user request."""
        request_lower = request.lower()
        
        # Extract directory path from request
        target_dir = self._extract_directory_path(request)
        
        # File Management capabilities
        if "file" in request_lower and ("size" in request_lower or "storage" in request_lower or "largest" in request_lower):
            return self._analyze_file_sizes(target_dir)
        
        # Code Analysis capabilities
        elif "code" in request_lower and ("analyze" in request_lower or "review" in request_lower):
            return self._analyze_code(target_dir)
        
        # Data Processing capabilities
        elif "data" in request_lower and ("process" in request_lower or "analyze" in request_lower):
            return self._process_data(target_dir)
        
        # Text Generation capabilities
        elif "text" in request_lower and ("generate" in request_lower or "write" in request_lower):
            return self._generate_text(request)
        
        # General help
        elif "help" in request_lower or "capabilities" in request_lower:
            return self._show_help()
        
        # Default response
        else:
            return f"🤖 {self.config.get('name', 'JARVIS')} here! I can help with:\n" + \
                   f"• File management and analysis\n" + \
                   f"• Code analysis and review\n" + \
                   f"• Data processing\n" + \
                   f"• Text generation\n\n" + \
                   f"Try asking me to 'analyze file sizes' or 'help' for more options!"
    
    def _extract_directory_path(self, request: str) -> Optional[str]:
        """Extract directory path from user request."""
        # Look for common directory patterns
        patterns = [
            r'in\s+([/\w\-\.~]+)',  # "in /path/to/dir"
            r'from\s+([/\w\-\.~]+)',  # "from /path/to/dir"
            r'at\s+([/\w\-\.~]+)',  # "at /path/to/dir"
            r'([/\w\-\.~]+)\s+directory',  # "/path/to/dir directory"
            r'([/\w\-\.~]+)\s+folder',  # "/path/to/dir folder"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                # Expand ~ to home directory
                if path.startswith('~'):
                    path = os.path.expanduser(path)
                # Check if path exists and is a directory
                if os.path.exists(path) and os.path.isdir(path):
                    return path
        
        return None
    
    def _analyze_file_sizes(self, target_dir: Optional[str] = None) -> str:
        """Analyze file sizes in the specified directory or current directory."""
        try:
            # Use target directory or current directory
            analysis_dir = target_dir if target_dir else os.getcwd()
            
            # Use du command to get directory sizes, sorted by size
            result = subprocess.run(
                ['du', '-h', '--max-depth=1', '.'],
                capture_output=True,
                text=True,
                cwd=analysis_dir
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Sort by size (largest first)
                lines.sort(key=lambda x: int(x.split('\t')[0]) if x.split('\t')[0].isdigit() else 0, reverse=True)
                
                dir_name = os.path.basename(analysis_dir) if analysis_dir != os.getcwd() else "current directory"
                response = f"📊 **File/Directory Size Analysis** in `{analysis_dir}` (largest first):\n\n"
                for line in lines[:20]:  # Show top 20
                    if line.strip():
                        size, path = line.split('\t', 1)
                        response += f"`{size:>8}` {path}\n"
                
                if len(lines) > 20:
                    response += f"\n... and {len(lines) - 20} more items\n"
                
                return response
            else:
                return f"❌ Error analyzing files in {analysis_dir}: {result.stderr}"
                
        except Exception as e:
            return f"❌ Error analyzing {analysis_dir if target_dir else 'current directory'}: {str(e)}"
    
    def _analyze_code(self, target_dir: Optional[str] = None) -> str:
        """Analyze code in the specified directory or current directory."""
        try:
            # Use target directory or current directory
            analysis_dir = target_dir if target_dir else os.getcwd()
            
            # Find Python files
            python_files = []
            for root, dirs, files in os.walk(analysis_dir):
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, file), analysis_dir)
                        python_files.append(rel_path)
            
            if not python_files:
                return f"📝 No Python files found in `{analysis_dir}`."
            
            response = f"🐍 **Code Analysis** in `{analysis_dir}` - Found {len(python_files)} Python files:\n\n"
            
            for py_file in python_files[:10]:  # Show first 10
                try:
                    full_path = os.path.join(analysis_dir, py_file)
                    with open(full_path, 'r') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        response += f"• `{py_file}` ({line_count} lines)\n"
                except:
                    response += f"• `{py_file}` (unreadable)\n"
            
            if len(python_files) > 10:
                response += f"\n... and {len(python_files) - 10} more Python files\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing code in {analysis_dir if target_dir else 'current directory'}: {str(e)}"
    
    def _process_data(self, target_dir: Optional[str] = None) -> str:
        """Process data files in the specified directory or current directory."""
        try:
            # Use target directory or current directory
            analysis_dir = target_dir if target_dir else os.getcwd()
            
            # Find common data files
            data_extensions = ['.csv', '.json', '.xlsx', '.txt', '.log']
            data_files = []
            
            for root, dirs, files in os.walk(analysis_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in data_extensions):
                        rel_path = os.path.relpath(os.path.join(root, file), analysis_dir)
                        data_files.append(rel_path)
            
            if not data_files:
                return f"📊 No data files found in `{analysis_dir}`."
            
            response = f"📊 **Data Files Found** in `{analysis_dir}` ({len(data_files)} files):\n\n"
            
            for data_file in data_files[:10]:  # Show first 10
                try:
                    full_path = os.path.join(analysis_dir, data_file)
                    size = os.path.getsize(full_path)
                    size_str = self._format_size(size)
                    response += f"• `{data_file}` ({size_str})\n"
                except:
                    response += f"• `{data_file}` (size unknown)\n"
            
            if len(data_files) > 10:
                response += f"\n... and {len(data_files) - 10} more data files\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error processing data in {analysis_dir if target_dir else 'current directory'}: {str(e)}"
    
    def _generate_text(self, request: str) -> str:
        """Generate text based on request."""
        return f"📝 **Text Generation**: I can help generate text! You asked: '{request}'\n\n" + \
               "For now, I'm a basic agent. To enable full AI text generation, " + \
               "I would need to be connected to the actual AI model specified in my config."
    
    def _show_help(self) -> str:
        """Show help information."""
        capabilities = self.config.get("capabilities", [])
        return f"🤖 **{self.config.get('name', 'JARVIS')} Help**\n\n" + \
               f"**My Capabilities:**\n" + \
               "\n".join([f"• {cap}" for cap in capabilities]) + \
               f"\n\n**Example Commands:**\n" + \
               f"• 'Analyze file sizes' - Show largest files/directories\n" + \
               f"• 'Analyze file sizes in /home/joshua' - Analyze specific directory\n" + \
               f"• 'Analyze code' - Review Python files\n" + \
               f"• 'Analyze code from ~/projects' - Analyze specific directory\n" + \
               f"• 'Process data' - List data files\n" + \
               f"• 'Process data at /var/log' - Process specific directory\n" + \
               f"• 'Generate text about...' - Text generation\n" + \
               f"• 'Help' - Show this help\n\n" + \
               f"**Directory Support:**\n" + \
               f"• Use 'in /path/to/dir', 'from /path/to/dir', or 'at /path/to/dir'\n" + \
               f"• Use ~ for home directory (e.g., 'in ~/Documents')\n" + \
               f"• If no directory specified, analyzes current directory\n\n" + \
               f"**Model:** {self.config.get('model', 'Unknown')}"
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def run_interactive(self):
        """Run the agent in interactive mode."""
        print(f"🤖 {self.config.get('name', 'Agent')} is ready!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                if user_input:
                    response = self.process_request(user_input)
                    print(f"Agent: {response}\n")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

def main():
    """Main entry point."""
    agent = JARVISAgent()
    agent.run_interactive()

if __name__ == "__main__":
    main()
