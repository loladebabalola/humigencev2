# inference/logging_utils.py

import os
import json
import time
import logging
import csv
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import threading
from collections import deque

console = Console()

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: str
    component: str
    message: str
    request_id: Optional[str] = None
    tenant_id: Optional[str] = None
    gpu_port: Optional[int] = None
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredLogger:
    """Structured logger for multi-tenant inference"""
    
    def __init__(self, log_dir: str = "logs", max_file_size: int = 100 * 1024 * 1024, 
                 max_files: int = 10, compression: bool = True):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compression = compression
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log files
        self.jsonl_file = self.log_dir / "inference.jsonl"
        self.csv_file = self.log_dir / "inference.csv"
        
        # In-memory buffer for batching
        self.buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()
        
        # Background thread for writing logs
        self.writer_thread = None
        self.running = False
        
        # Initialize CSV file with headers
        self.init_csv_file()
        
        # Start background writer
        self.start_writer()
    
    def init_csv_file(self):
        """Initialize CSV file with headers"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'level', 'component', 'message', 'request_id',
                    'tenant_id', 'gpu_port', 'tokens', 'latency_ms', 'error_code'
                ])
    
    def start_writer(self):
        """Start background log writer thread"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def stop_writer(self):
        """Stop background log writer thread"""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
    
    def _writer_loop(self):
        """Background thread for writing logs"""
        while self.running:
            try:
                # Process buffer
                with self.buffer_lock:
                    if self.buffer:
                        entries = list(self.buffer)
                        self.buffer.clear()
                    else:
                        entries = []
                
                if entries:
                    self._write_entries(entries)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"[red]❌ Log writer error: {e}[/red]")
                time.sleep(1.0)
    
    def _write_entries(self, entries: List[LogEntry]):
        """Write log entries to files"""
        try:
            # Write to JSONL file
            with open(self.jsonl_file, 'a') as f:
                for entry in entries:
                    f.write(json.dumps(asdict(entry), default=str) + '\n')
            
            # Write to CSV file
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for entry in entries:
                    writer.writerow([
                        entry.timestamp,
                        entry.level,
                        entry.component,
                        entry.message,
                        entry.request_id or '',
                        entry.tenant_id or '',
                        entry.gpu_port or '',
                        entry.tokens or '',
                        entry.latency_ms or '',
                        entry.error_code or ''
                    ])
            
            # Check if rotation is needed
            self._check_rotation()
            
        except Exception as e:
            print(f"[red]❌ Failed to write log entries: {e}[/red]")
    
    def _check_rotation(self):
        """Check if log rotation is needed"""
        try:
            if self.jsonl_file.exists() and self.jsonl_file.stat().st_size > self.max_file_size:
                self._rotate_logs()
        except Exception as e:
            print(f"[red]❌ Log rotation error: {e}[/red]")
    
    def _rotate_logs(self):
        """Rotate log files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Rotate JSONL file
            if self.jsonl_file.exists():
                rotated_file = self.log_dir / f"inference_{timestamp}.jsonl"
                shutil.move(str(self.jsonl_file), str(rotated_file))
                
                # Compress if enabled
                if self.compression:
                    with open(rotated_file, 'rb') as f_in:
                        with gzip.open(f"{rotated_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    rotated_file.unlink()  # Remove uncompressed file
            
            # Rotate CSV file
            if self.csv_file.exists():
                rotated_file = self.log_dir / f"inference_{timestamp}.csv"
                shutil.move(str(self.csv_file), str(rotated_file))
                
                # Compress if enabled
                if self.compression:
                    with open(rotated_file, 'rb') as f_in:
                        with gzip.open(f"{rotated_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    rotated_file.unlink()  # Remove uncompressed file
            
            # Clean up old files
            self._cleanup_old_files()
            
            # Reinitialize CSV file
            self.init_csv_file()
            
            print(f"[blue]📁 Log files rotated: {timestamp}[/blue]")
            
        except Exception as e:
            print(f"[red]❌ Log rotation failed: {e}[/red]")
    
    def _cleanup_old_files(self):
        """Clean up old log files"""
        try:
            # Get all log files
            log_files = list(self.log_dir.glob("inference_*.jsonl*")) + list(self.log_dir.glob("inference_*.csv*"))
            
            # Sort by modification time
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent files
            for old_file in log_files[self.max_files:]:
                old_file.unlink()
                print(f"[blue]🗑️ Removed old log file: {old_file.name}[/blue]")
                
        except Exception as e:
            print(f"[red]❌ Cleanup error: {e}[/red]")
    
    def log(self, level: str, component: str, message: str, **kwargs):
        """Log a structured entry"""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            component=component,
            message=message,
            **kwargs
        )
        
        with self.buffer_lock:
            self.buffer.append(entry)
    
    def info(self, component: str, message: str, **kwargs):
        """Log info level message"""
        self.log("INFO", component, message, **kwargs)
    
    def warning(self, component: str, message: str, **kwargs):
        """Log warning level message"""
        self.log("WARNING", component, message, **kwargs)
    
    def error(self, component: str, message: str, **kwargs):
        """Log error level message"""
        self.log("ERROR", component, message, **kwargs)
    
    def debug(self, component: str, message: str, **kwargs):
        """Log debug level message"""
        self.log("DEBUG", component, message, **kwargs)
    
    def log_request(self, request_id: str, tenant_id: str, gpu_port: int, 
                   tokens: int, latency_ms: float, success: bool = True, 
                   error_code: Optional[str] = None):
        """Log a request with metrics"""
        level = "INFO" if success else "ERROR"
        message = f"Request processed" if success else f"Request failed: {error_code}"
        
        self.log(
            level=level,
            component="router",
            message=message,
            request_id=request_id,
            tenant_id=tenant_id,
            gpu_port=gpu_port,
            tokens=tokens,
            latency_ms=latency_ms,
            error_code=error_code
        )
    
    def log_instance_event(self, instance_id: int, event: str, message: str, **kwargs):
        """Log an instance event"""
        self.log(
            level="INFO",
            component=f"instance_{instance_id}",
            message=f"{event}: {message}",
            **kwargs
        )
    
    def log_tenant_event(self, tenant_id: str, event: str, message: str, **kwargs):
        """Log a tenant event"""
        self.log(
            level="INFO",
            component="tenant_manager",
            message=f"Tenant {tenant_id} - {event}: {message}",
            tenant_id=tenant_id,
            **kwargs
        )
    
    def get_recent_logs(self, hours: int = 1, level: Optional[str] = None, 
                       component: Optional[str] = None) -> List[LogEntry]:
        """Get recent log entries"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            entries = []
            
            if self.jsonl_file.exists():
                with open(self.jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry = LogEntry(**data)
                            
                            # Filter by time
                            if entry.timestamp < cutoff_time:
                                continue
                            
                            # Filter by level
                            if level and entry.level != level:
                                continue
                            
                            # Filter by component
                            if component and entry.component != component:
                                continue
                            
                            entries.append(entry)
                            
                        except Exception:
                            continue  # Skip malformed entries
            
            return entries
            
        except Exception as e:
            print(f"[red]❌ Failed to get recent logs: {e}[/red]")
            return []
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary from logs"""
        try:
            entries = self.get_recent_logs(hours)
            
            # Filter request logs
            request_entries = [e for e in entries if e.request_id and e.tokens is not None]
            
            if not request_entries:
                return {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'total_tokens': 0,
                    'avg_latency_ms': 0,
                    'requests_per_hour': 0,
                    'tokens_per_hour': 0,
                    'error_rate': 0
                }
            
            total_requests = len(request_entries)
            successful_requests = len([e for e in request_entries if e.level == "INFO"])
            failed_requests = len([e for e in request_entries if e.level == "ERROR"])
            total_tokens = sum(e.tokens for e in request_entries)
            
            latencies = [e.latency_ms for e in request_entries if e.latency_ms is not None]
            avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
            
            requests_per_hour = total_requests / hours
            tokens_per_hour = total_tokens / hours
            error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'total_tokens': total_tokens,
                'avg_latency_ms': avg_latency_ms,
                'requests_per_hour': requests_per_hour,
                'tokens_per_hour': tokens_per_hour,
                'error_rate': error_rate
            }
            
        except Exception as e:
            print(f"[red]❌ Failed to get metrics summary: {e}[/red]")
            return {}
    
    def display_recent_logs(self, hours: int = 1, limit: int = 50):
        """Display recent logs in a table"""
        entries = self.get_recent_logs(hours)[-limit:]  # Get last N entries
        
        if not entries:
            console.print("[yellow]⚠️ No recent logs found[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title=f"Recent Logs (Last {hours}h)")
        table.add_column("Time", style="cyan", width=20)
        table.add_column("Level", style="white", width=8)
        table.add_column("Component", style="blue", width=15)
        table.add_column("Message", style="white", width=40)
        table.add_column("Request ID", style="green", width=12)
        table.add_column("Tenant", style="yellow", width=10)
        
        for entry in entries:
            time_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            level_color = {
                "INFO": "green",
                "WARNING": "yellow", 
                "ERROR": "red",
                "DEBUG": "blue"
            }.get(entry.level, "white")
            
            table.add_row(
                time_str,
                f"[{level_color}]{entry.level}[/{level_color}]",
                entry.component,
                entry.message[:40] + "..." if len(entry.message) > 40 else entry.message,
                entry.request_id or "",
                entry.tenant_id or ""
            )
        
        console.print(table)
    
    def display_metrics_summary(self, hours: int = 1):
        """Display metrics summary"""
        metrics = self.get_metrics_summary(hours)
        
        if not metrics:
            console.print("[yellow]⚠️ No metrics available[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title=f"Metrics Summary (Last {hours}h)")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=15)
        table.add_column("Unit", style="green", width=10)
        
        table.add_row("Total Requests", f"{metrics['total_requests']:,}", "requests")
        table.add_row("Successful", f"{metrics['successful_requests']:,}", "requests")
        table.add_row("Failed", f"{metrics['failed_requests']:,}", "requests")
        table.add_row("Error Rate", f"{metrics['error_rate']:.2f}", "%")
        table.add_row("Total Tokens", f"{metrics['total_tokens']:,}", "tokens")
        table.add_row("Avg Latency", f"{metrics['avg_latency_ms']:.2f}", "ms")
        table.add_row("Requests/Hour", f"{metrics['requests_per_hour']:.1f}", "req/h")
        table.add_row("Tokens/Hour", f"{metrics['tokens_per_hour']:.1f}", "tokens/h")
        
        console.print(table)
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up logs older than specified days"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            removed_count = 0
            
            for log_file in self.log_dir.glob("inference_*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            
            console.print(f"[green]✅ Cleaned up {removed_count} old log files[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Cleanup failed: {e}[/red]")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_writer()

# Global logger instance
_logger = None

def get_logger() -> StructuredLogger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger

def log_info(component: str, message: str, **kwargs):
    """Log info message using global logger"""
    get_logger().info(component, message, **kwargs)

def log_warning(component: str, message: str, **kwargs):
    """Log warning message using global logger"""
    get_logger().warning(component, message, **kwargs)

def log_error(component: str, message: str, **kwargs):
    """Log error message using global logger"""
    get_logger().error(component, message, **kwargs)

def log_debug(component: str, message: str, **kwargs):
    """Log debug message using global logger"""
    get_logger().debug(component, message, **kwargs)

def log_request(request_id: str, tenant_id: str, gpu_port: int, 
               tokens: int, latency_ms: float, success: bool = True, 
               error_code: Optional[str] = None):
    """Log request using global logger"""
    get_logger().log_request(request_id, tenant_id, gpu_port, tokens, latency_ms, success, error_code)

def main():
    """CLI for log management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Tenant Inference Log Manager")
    parser.add_argument("--action", choices=["view", "metrics", "cleanup"], default="view",
                       help="Action to perform")
    parser.add_argument("--hours", type=int, default=1, help="Hours of logs to analyze")
    parser.add_argument("--days", type=int, default=7, help="Days of logs to keep during cleanup")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of log entries to display")
    
    args = parser.parse_args()
    
    logger = StructuredLogger()
    
    if args.action == "view":
        logger.display_recent_logs(args.hours, args.limit)
    elif args.action == "metrics":
        logger.display_metrics_summary(args.hours)
    elif args.action == "cleanup":
        logger.cleanup_old_logs(args.days)

if __name__ == "__main__":
    main()
