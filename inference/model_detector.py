# inference/model_detector.py

import os
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass
class ModelInfo:
    """Model information container"""
    name: str
    path: str
    size_gb: float
    context_window: int
    quantization: str
    architecture: str
    is_gguf: bool
    last_modified: float

class ModelDetector:
    """Auto-detect models and parse metadata"""
    
    def __init__(self, models_dir: str = "~/models"):
        self.models_dir = Path(models_dir).expanduser()
        self.models: List[ModelInfo] = []
        self.default_model: Optional[ModelInfo] = None
        
    def scan_models(self) -> Tuple[int, List[ModelInfo]]:
        """Scan for available models in the models directory"""
        self.models = []
        
        if not self.models_dir.exists():
            console.print(f"[yellow]⚠️ Models directory not found: {self.models_dir}[/yellow]")
            console.print("[blue]💡 Create the directory and add your GGUF models[/blue]")
            return 0, []
        
        # Find all GGUF files
        gguf_files = list(self.models_dir.rglob("*.gguf"))
        
        if not gguf_files:
            console.print("[yellow]⚠️ No GGUF files found[/yellow]")
            return 0, []
        
        for model_path in gguf_files:
            try:
                model_info = self._analyze_model(model_path)
                if model_info:
                    self.models.append(model_info)
            except Exception as e:
                console.print(f"[red]❌ Error analyzing {model_path.name}: {e}[/red]")
                continue
        
        # Sort by last modified (newest first)
        self.models.sort(key=lambda x: x.last_modified, reverse=True)
        
        # Set default model (most recently modified)
        if self.models:
            self.default_model = self.models[0]
        
        return len(self.models), self.models
    
    def _analyze_model(self, model_path: Path) -> Optional[ModelInfo]:
        """Analyze a single model file"""
        try:
            # Get file info
            stat = model_path.stat()
            size_gb = stat.st_size / (1024**3)
            last_modified = stat.st_mtime
            
            # Extract name from path
            name = model_path.stem
            
            # Try to parse GGUF metadata
            context_window, quantization, architecture = self._parse_gguf_metadata(model_path)
            
            model_info = ModelInfo(
                name=name,
                path=str(model_path),
                size_gb=size_gb,
                context_window=context_window,
                quantization=quantization,
                architecture=architecture,
                is_gguf=True,
                last_modified=last_modified
            )
            
            return model_info
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing {model_path}: {e}[/red]")
            return None
    
    def _parse_gguf_metadata(self, model_path: Path) -> Tuple[int, str, str]:
        """Parse GGUF file metadata to extract context window and other info"""
        try:
            # Try using llama-cpp-python if available
            try:
                import llama_cpp
                from llama_cpp import Llama
                
                # Try to load model metadata without loading the full model
                model = Llama(model_path=str(model_path), verbose=False)
                context_window = getattr(model, 'n_ctx', 131072)
                # Ensure it's an integer, not a method
                if callable(context_window):
                    context_window = context_window()
                quantization = "llama_cpp"
                architecture = "llama"
                
                # Clean up
                del model
                return int(context_window), quantization, architecture
                
            except ImportError:
                # Fallback to manual parsing
                pass
            except Exception:
                # If llama-cpp fails, fall back to manual parsing
                pass
            
            # Manual GGUF parsing
            with open(model_path, 'rb') as f:
                # Read GGUF magic
                magic = f.read(4)
                if magic != b'GGUF':
                    return 131072, "unknown", "unknown"
                
                # Read version
                version = struct.unpack('<I', f.read(4))[0]
                
                # Read tensor count
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                
                # Read key-value count
                kv_count = struct.unpack('<Q', f.read(8))[0]
                
                # Parse key-value pairs
                context_window = 131072  # Default
                quantization = "unknown"
                architecture = "unknown"
                
                for _ in range(kv_count):
                    try:
                        # Read key length and key
                        key_len = struct.unpack('<Q', f.read(8))[0]
                        key = f.read(key_len).decode('utf-8')
                        
                        # Read value type
                        value_type = struct.unpack('<I', f.read(4))[0]
                        
                        # Read value based on type
                        if value_type == 4:  # String
                            value_len = struct.unpack('<Q', f.read(8))[0]
                            value = f.read(value_len).decode('utf-8')
                            
                            if key == "llama.context_length":
                                try:
                                    context_window = int(value)
                                except:
                                    pass
                            elif key == "general.architecture":
                                architecture = value
                            elif key == "general.quantization_version":
                                quantization = value
                        
                        elif value_type == 5:  # UInt32
                            value = struct.unpack('<I', f.read(4))[0]
                            if key == "llama.context_length":
                                context_window = value
                        
                        elif value_type == 6:  # UInt64
                            value = struct.unpack('<Q', f.read(8))[0]
                            if key == "llama.context_length":
                                context_window = value
                        
                        else:
                            # Skip other types
                            if value_type == 0:  # UInt8
                                f.read(1)
                            elif value_type == 1:  # Int8
                                f.read(1)
                            elif value_type == 2:  # UInt16
                                f.read(2)
                            elif value_type == 3:  # Int16
                                f.read(2)
                            elif value_type == 7:  # Float32
                                f.read(4)
                            elif value_type == 8:  # Float64
                                f.read(8)
                            elif value_type == 9:  # Bool
                                f.read(1)
                            else:
                                # Unknown type, skip
                                pass
                    except (struct.error, UnicodeDecodeError, ValueError):
                        # Skip malformed entries
                        continue
                
                return context_window, quantization, architecture
                
        except Exception as e:
            # Always return safe fallback values instead of crashing
            console.print(f"[dim]Could not parse GGUF metadata for {model_path.name}: {str(e)[:50]}...[/dim]")
            return 131072, "unknown", "unknown"
    
    def get_default_model(self) -> Optional[ModelInfo]:
        """Get the default model (most recently modified)"""
        return self.default_model
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model by name"""
        for model in self.models:
            if model.name == name:
                return model
        return None
    
    def get_model_by_path(self, path: str) -> Optional[ModelInfo]:
        """Get model by path"""
        for model in self.models:
            if model.path == path:
                return model
        return None
    
    def display_models(self):
        """Display a table of detected models"""
        if not self.models:
            console.print("[yellow]⚠️ No models detected[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Detected Models")
        table.add_column("Name", style="cyan", width=25)
        table.add_column("Size", style="green", width=10)
        table.add_column("Context", style="blue", width=10)
        table.add_column("Quantization", style="yellow", width=12)
        table.add_column("Architecture", style="magenta", width=15)
        table.add_column("Path", style="white", width=40)
        
        for model in self.models:
            is_default = "⭐" if model == self.default_model else ""
            table.add_row(
                f"{model.name}{is_default}",
                f"{model.size_gb:.1f}GB",
                f"{model.context_window:,}" if isinstance(model.context_window, (int, float)) else str(model.context_window),
                model.quantization,
                model.architecture,
                model.path
            )
        
        console.print(table)
    
    def validate_model(self, model_path: str) -> bool:
        """Validate that a model file exists and is accessible"""
        path = Path(model_path)
        
        if not path.exists():
            console.print(f"[red]❌ Model file not found: {model_path}[/red]")
            return False
        
        if not path.suffix.lower() == '.gguf':
            console.print(f"[red]❌ Model file is not a GGUF file: {model_path}[/red]")
            return False
        
        # Check file size (should be at least 100MB)
        size_gb = path.stat().st_size / (1024**3)
        if size_gb < 0.1:
            console.print(f"[red]❌ Model file too small: {size_gb:.2f}GB[/red]")
            return False
        
        console.print(f"[green]✅ Model validated: {path.name} ({size_gb:.1f}GB)[/green]")
        return True
    
    def get_recommended_context_window(self, model: ModelInfo) -> int:
        """Get recommended context window based on model size and type"""
        if model.context_window > 0:
            return model.context_window
        
        # Fallback recommendations based on model size
        if model.size_gb < 4:
            return 4096
        elif model.size_gb < 8:
            return 8192
        elif model.size_gb < 16:
            return 16384
        else:
            return 32768
    
    def get_system_summary(self) -> Dict:
        """Get a complete system summary for the summary panel"""
        return {
            "model_count": len(self.models),
            "default_model": {
                "name": self.default_model.name if self.default_model else None,
                "path": self.default_model.path if self.default_model else None,
                "size_gb": self.default_model.size_gb if self.default_model else 0,
                "context_window": self.default_model.context_window if self.default_model else 0,
                "quantization": self.default_model.quantization if self.default_model else "unknown"
            } if self.default_model else None,
            "models": [
                {
                    "name": model.name,
                    "path": model.path,
                    "size_gb": model.size_gb,
                    "context_window": model.context_window,
                    "quantization": model.quantization,
                    "architecture": model.architecture,
                    "is_default": model == self.default_model
                }
                for model in self.models
            ]
        }
    
    def create_model_config(self, model: ModelInfo) -> Dict:
        """Create a model configuration for the inference system"""
        return {
            "model_path": model.path,
            "context_window": model.context_window,
            "quantization": model.quantization,
            "architecture": model.architecture,
            "size_gb": model.size_gb
        }

def main():
    """Test model detection"""
    detector = ModelDetector()
    
    console.print("[bold cyan]Model Detection Test[/bold cyan]")
    console.print("=" * 50)
    
    # Scan for models
    count, models = detector.scan_models()
    
    if count > 0:
        # Display models
        detector.display_models()
        
        # Show default model
        default = detector.get_default_model()
        if default:
            console.print(f"\n[blue]Default Model:[/blue]")
            console.print(f"  Name: {default.name}")
            console.print(f"  Path: {default.path}")
            console.print(f"  Size: {default.size_gb:.1f}GB")
            console.print(f"  Context: {default.context_window:,}")
            console.print(f"  Quantization: {default.quantization}")
        
        # Show system summary
        summary = detector.get_system_summary()
        console.print(f"\n[blue]System Summary:[/blue]")
        console.print(f"  Models: {summary['model_count']} detected")
        if summary['default_model']:
            console.print(f"  Default: {summary['default_model']['name']}")
    else:
        console.print("[red]❌ No models detected - cannot proceed[/red]")
        console.print("[blue]💡 Add GGUF models to ~/models/ directory[/blue]")

if __name__ == "__main__":
    main()
