# inference/gpu_detector.py

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class GPUInfo:
    """GPU information container"""
    index: int
    name: str
    memory_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_c: float
    is_available: bool

class GPUDetector:
    """Auto-detect GPUs and create port mappings"""
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self.port_mapping: Dict[int, int] = {}
        self.base_port = 8000
        
    def detect_gpus(self) -> Tuple[int, List[GPUInfo]]:
        """Detect available GPUs and return count + info"""
        self.gpus = []
        
        try:
            # Lazy import to avoid CUDA warnings at startup
            import torch
            
            if not torch.cuda.is_available():
                console.print("[yellow]⚠️ CUDA not available - no GPUs detected[/yellow]")
                return 0, []
            
            gpu_count = torch.cuda.device_count()
            console.print(f"[blue]🔍 Detecting {gpu_count} GPU(s)...[/blue]")
            
            for i in range(gpu_count):
                try:
                    # Get basic GPU info
                    props = torch.cuda.get_device_properties(i)
                    name = props.name
                    total_memory = props.total_memory / (1024**3)  # Convert to GB
                    
                    # Get current memory usage
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_used = memory_allocated + memory_reserved
                    memory_free = total_memory - memory_used
                    
                    # Get utilization (approximate)
                    utilization = (memory_used / total_memory) * 100
                    
                    gpu_info = GPUInfo(
                        index=i,
                        name=name,
                        memory_gb=total_memory,
                        memory_used_gb=memory_used,
                        memory_free_gb=memory_free,
                        utilization_percent=utilization,
                        temperature_c=0.0,  # Would need nvidia-ml-py for real temp
                        is_available=True
                    )
                    
                    self.gpus.append(gpu_info)
                    
                except Exception as e:
                    console.print(f"[red]❌ Error detecting GPU {i}: {e}[/red]")
                    continue
            
            console.print(f"[green]✅ Detected {len(self.gpus)} GPU(s)[/green]")
            return len(self.gpus), self.gpus
            
        except Exception as e:
            console.print(f"[red]❌ GPU detection failed: {e}[/red]")
            return 0, []
    
    def create_port_mapping(self, base_port: int = 8000) -> Dict[int, int]:
        """Create automatic port mapping for detected GPUs"""
        self.base_port = base_port
        self.port_mapping = {}
        
        for i, gpu in enumerate(self.gpus):
            port = base_port + i
            self.port_mapping[gpu.index] = port
            console.print(f"[blue]🔗 GPU {gpu.index} → Port {port}[/blue]")
        
        return self.port_mapping
    
    def get_gpu_port_map(self) -> Dict[int, int]:
        """Get the current GPU to port mapping"""
        return self.port_mapping
    
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs"""
        return [gpu for gpu in self.gpus if gpu.is_available]
    
    def get_gpu_by_index(self, index: int) -> Optional[GPUInfo]:
        """Get GPU info by index"""
        for gpu in self.gpus:
            if gpu.index == index:
                return gpu
        return None
    
    def display_gpu_summary(self):
        """Display a summary table of detected GPUs"""
        if not self.gpus:
            console.print("[yellow]⚠️ No GPUs detected[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="GPU Detection Summary")
        table.add_column("Index", style="cyan", width=6)
        table.add_column("Name", style="white", width=40)
        table.add_column("Total Memory", style="green", width=12)
        table.add_column("Free Memory", style="blue", width=12)
        table.add_column("Utilization", style="yellow", width=12)
        table.add_column("Port", style="magenta", width=8)
        table.add_column("Status", style="white", width=10)
        
        for gpu in self.gpus:
            port = self.port_mapping.get(gpu.index, "N/A")
            status = "🟢 Available" if gpu.is_available else "🔴 Unavailable"
            
            table.add_row(
                str(gpu.index),
                gpu.name,
                f"{gpu.memory_gb:.1f}GB",
                f"{gpu.memory_free_gb:.1f}GB",
                f"{gpu.utilization_percent:.1f}%",
                str(port),
                status
            )
        
        console.print(table)
    
    def validate_gpu_setup(self) -> bool:
        """Validate that GPU setup is ready for inference"""
        if not self.gpus:
            console.print("[red]❌ No GPUs available for inference[/red]")
            return False
        
        available_gpus = self.get_available_gpus()
        if not available_gpus:
            console.print("[red]❌ No available GPUs for inference[/red]")
            return False
        
        # Check if we have enough free memory (at least 4GB per GPU)
        for gpu in available_gpus:
            if gpu.memory_free_gb < 4.0:
                console.print(f"[yellow]⚠️ GPU {gpu.index} has low memory: {gpu.memory_free_gb:.1f}GB free[/yellow]")
        
        console.print(f"[green]✅ GPU setup validated - {len(available_gpus)} GPU(s) ready[/green]")
        return True
    
    def get_recommended_parallel_slots(self, gpu_index: int) -> int:
        """Get recommended parallel slots based on GPU memory"""
        gpu = self.get_gpu_by_index(gpu_index)
        if not gpu:
            return 1
        
        # Rough estimation: 1 slot per 4GB of free memory
        # Minimum 1, maximum 8
        recommended = max(1, min(8, int(gpu.memory_free_gb / 4)))
        return recommended
    
    def get_system_summary(self) -> Dict:
        """Get a complete system summary for the summary panel"""
        return {
            "gpu_count": len(self.gpus),
            "available_gpus": len(self.get_available_gpus()),
            "port_mapping": self.port_mapping,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_gb": gpu.memory_gb,
                    "memory_free_gb": gpu.memory_free_gb,
                    "utilization_percent": gpu.utilization_percent,
                    "port": self.port_mapping.get(gpu.index),
                    "is_available": gpu.is_available
                }
                for gpu in self.gpus
            ]
        }

def main():
    """Test GPU detection"""
    detector = GPUDetector()
    
    console.print("[bold cyan]GPU Detection Test[/bold cyan]")
    console.print("=" * 50)
    
    # Detect GPUs
    count, gpus = detector.detect_gpus()
    
    if count > 0:
        # Create port mapping
        detector.create_port_mapping()
        
        # Display summary
        detector.display_gpu_summary()
        
        # Validate setup
        detector.validate_gpu_setup()
        
        # Show system summary
        summary = detector.get_system_summary()
        console.print(f"\n[blue]System Summary:[/blue]")
        console.print(f"  GPUs: {summary['gpu_count']} detected, {summary['available_gpus']} available")
        console.print(f"  Ports: {list(summary['port_mapping'].values())}")
    else:
        console.print("[red]❌ No GPUs detected - cannot proceed[/red]")

if __name__ == "__main__":
    main()
