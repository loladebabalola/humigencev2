# inference/loadgen.py

import asyncio
import aiohttp
import time
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import statistics

console = Console()

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    scenario: str
    duration: int  # seconds
    concurrent_users: int
    requests_per_second: float
    base_url: str = "http://localhost:8000"
    tenant_id: str = "default"

@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: str
    tenant_id: str
    success: bool
    status_code: int
    response_time: float
    tokens_generated: int
    error_message: Optional[str] = None
    timestamp: float = 0.0

@dataclass
class LoadTestResult:
    """Results of a load test"""
    config: LoadTestConfig
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_duration: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    tokens_per_second: float
    error_rate: float
    results: List[RequestResult]

class LoadGenerator:
    """Load generator for multi-tenant inference testing"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.start_time = 0.0
        self.end_time = 0.0
        
        # Test prompts for different scenarios
        self.prompts = {
            "burst": [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about AI.",
                "What are the benefits of renewable energy?"
            ],
            "mixed": [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about AI.",
                "What are the benefits of renewable energy?",
                "Write a comprehensive analysis of the impact of artificial intelligence on modern society, including its benefits, challenges, and future implications for various industries such as healthcare, education, transportation, and manufacturing.",
                "Create a detailed technical specification for a distributed microservices architecture that can handle 1 million concurrent users, including database design, caching strategies, load balancing, monitoring, and security considerations.",
                "Analyze the philosophical implications of consciousness in artificial intelligence systems, exploring questions about machine sentience, the nature of mind, ethical considerations, and the potential for AI to develop genuine understanding and creativity."
            ],
            "soak": [
                "Generate a creative story about a robot learning to paint.",
                "Explain the concept of machine learning to a 10-year-old.",
                "What are the key principles of good software design?",
                "Describe the process of photosynthesis in plants.",
                "How does blockchain technology work?",
                "What are the main causes of climate change?",
                "Explain the theory of relativity in simple terms.",
                "What are the benefits of regular exercise?",
                "How do computers process information?",
                "What is the importance of biodiversity?"
            ]
        }
    
    async def run_test(self) -> LoadTestResult:
        """Run the load test"""
        console.print(f"[bold green]🧪 Starting Load Test: {self.config.scenario}[/bold green]")
        console.print(f"⏱️ Duration: {self.config.duration}s")
        console.print(f"👥 Concurrent Users: {self.config.concurrent_users}")
        console.print(f"📊 Target RPS: {self.config.requests_per_second}")
        
        self.start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        # Create tasks for concurrent users
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self.user_worker(user_id, semaphore)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.duration
            )
        except asyncio.TimeoutError:
            console.print("[yellow]⚠️ Test duration reached, stopping...[/yellow]")
        
        self.end_time = time.time()
        
        # Calculate results
        return self.calculate_results()
    
    async def user_worker(self, user_id: int, semaphore: asyncio.Semaphore):
        """Worker function for a single user"""
        async with semaphore:
            while time.time() - self.start_time < self.config.duration:
                try:
                    # Calculate delay between requests
                    delay = 1.0 / self.config.requests_per_second
                    await asyncio.sleep(delay)
                    
                    # Make request
                    result = await self.make_request(user_id)
                    self.results.append(result)
                    
                except Exception as e:
                    console.print(f"[red]❌ User {user_id} error: {e}[/red]")
                    await asyncio.sleep(1.0)  # Brief pause on error
    
    async def make_request(self, user_id: int) -> RequestResult:
        """Make a single request to the inference API"""
        request_id = f"user_{user_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Select prompt based on scenario
        prompt = self.select_prompt()
        
        # Prepare request
        request_data = {
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "user": self.config.tenant_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    response_time = time.time() - start_time
                    response_data = await response.json()
                    
                    tokens_generated = response_data.get('usage', {}).get('completion_tokens', 0)
                    
                    return RequestResult(
                        request_id=request_id,
                        tenant_id=self.config.tenant_id,
                        success=response.status == 200,
                        status_code=response.status,
                        response_time=response_time,
                        tokens_generated=tokens_generated,
                        timestamp=time.time()
                    )
                    
        except asyncio.TimeoutError:
            return RequestResult(
                request_id=request_id,
                tenant_id=self.config.tenant_id,
                success=False,
                status_code=408,
                response_time=time.time() - start_time,
                tokens_generated=0,
                error_message="Request timeout",
                timestamp=time.time()
            )
        except Exception as e:
            return RequestResult(
                request_id=request_id,
                tenant_id=self.config.tenant_id,
                success=False,
                status_code=500,
                response_time=time.time() - start_time,
                tokens_generated=0,
                error_message=str(e),
                timestamp=time.time()
            )
    
    def select_prompt(self) -> str:
        """Select a prompt based on the test scenario"""
        prompts = self.prompts.get(self.config.scenario, self.prompts["burst"])
        return random.choice(prompts)
    
    def calculate_results(self) -> LoadTestResult:
        """Calculate test results from collected data"""
        if not self.results:
            return LoadTestResult(
                config=self.config,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_tokens=0,
                total_duration=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                tokens_per_second=0,
                error_rate=0,
                results=[]
            )
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        total_requests = len(self.results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        
        total_tokens = sum(r.tokens_generated for r in self.results)
        total_duration = self.end_time - self.start_time
        
        response_times = [r.response_time for r in self.results]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p50_response_time = statistics.median(response_times) if response_times else 0
        p95_response_time = self.percentile(response_times, 95) if response_times else 0
        p99_response_time = self.percentile(response_times, 99) if response_times else 0
        
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        return LoadTestResult(
            config=self.config,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens,
            total_duration=total_duration,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            error_rate=error_rate,
            results=self.results
        )
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def save_results(self, result: LoadTestResult, output_dir: Path):
        """Save test results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        result_dict = asdict(result)
        result_dict['config'] = asdict(result.config)
        
        with open(output_dir / "result.json", 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save CSV report
        with open(output_dir / "report.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'request_id', 'tenant_id', 'success', 'status_code',
                'response_time', 'tokens_generated', 'error_message', 'timestamp'
            ])
            
            for req_result in result.results:
                writer.writerow([
                    req_result.request_id,
                    req_result.tenant_id,
                    req_result.success,
                    req_result.status_code,
                    req_result.response_time,
                    req_result.tokens_generated,
                    req_result.error_message or '',
                    req_result.timestamp
                ])
        
        console.print(f"[green]✅ Results saved to {output_dir}[/green]")
    
    def display_results(self, result: LoadTestResult):
        """Display test results in a formatted table"""
        console.print("\n📊 [bold]Load Test Results[/bold]")
        console.print("="*60)
        
        # Summary table
        summary_table = Table(show_header=True, box=None)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="white", width=15)
        summary_table.add_column("Unit", style="green", width=10)
        
        summary_table.add_row("Total Requests", f"{result.total_requests:,}", "requests")
        summary_table.add_row("Successful", f"{result.successful_requests:,}", "requests")
        summary_table.add_row("Failed", f"{result.failed_requests:,}", "requests")
        summary_table.add_row("Error Rate", f"{result.error_rate:.2f}", "%")
        summary_table.add_row("Total Tokens", f"{result.total_tokens:,}", "tokens")
        summary_table.add_row("Duration", f"{result.total_duration:.2f}", "seconds")
        summary_table.add_row("RPS", f"{result.requests_per_second:.2f}", "req/s")
        summary_table.add_row("Tokens/sec", f"{result.tokens_per_second:.2f}", "tokens/s")
        
        console.print(summary_table)
        
        # Response time table
        console.print("\n⏱️ [bold]Response Time Statistics[/bold]")
        response_table = Table(show_header=True, box=None)
        response_table.add_column("Metric", style="cyan", width=20)
        response_table.add_column("Value", style="white", width=15)
        response_table.add_column("Unit", style="green", width=10)
        
        response_table.add_row("Average", f"{result.avg_response_time:.3f}", "seconds")
        response_table.add_row("P50 (Median)", f"{result.p50_response_time:.3f}", "seconds")
        response_table.add_row("P95", f"{result.p95_response_time:.3f}", "seconds")
        response_table.add_row("P99", f"{result.p99_response_time:.3f}", "seconds")
        
        console.print(response_table)
        
        # Error analysis
        if result.failed_requests > 0:
            console.print("\n❌ [bold]Error Analysis[/bold]")
            error_counts = {}
            for req in result.results:
                if not req.success:
                    error_key = f"{req.status_code}: {req.error_message or 'Unknown error'}"
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            error_table = Table(show_header=True, box=None)
            error_table.add_column("Error", style="red", width=40)
            error_table.add_column("Count", style="white", width=10)
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                error_table.add_row(error, str(count))
            
            console.print(error_table)

def run_load_test(scenario: str, duration: int = 60, concurrent_users: int = 10, 
                 requests_per_second: float = 5.0, base_url: str = "http://localhost:8000",
                 tenant_id: str = "default", output_dir: Optional[str] = None):
    """Run a load test with the specified parameters"""
    
    config = LoadTestConfig(
        scenario=scenario,
        duration=duration,
        concurrent_users=concurrent_users,
        requests_per_second=requests_per_second,
        base_url=base_url,
        tenant_id=tenant_id
    )
    
    generator = LoadGenerator(config)
    
    # Run the test
    result = asyncio.run(generator.run_test())
    
    # Display results
    generator.display_results(result)
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        generator.save_results(result, output_path)
    else:
        # Default output directory
        timestamp = int(time.time())
        output_path = Path(f"runs/load_test_{scenario}_{timestamp}")
        generator.save_results(result, output_path)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Multi-Tenant Inference Load Generator")
    parser.add_argument("--scenario", choices=["burst", "mixed", "soak"], default="burst",
                       help="Test scenario")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--concurrent-users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--rps", type=float, default=5.0, help="Target requests per second")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the inference API")
    parser.add_argument("--tenant-id", default="default", help="Tenant ID for requests")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    console.print(f"[bold green]🧪 Multi-Tenant Inference Load Generator[/bold green]")
    console.print(f"📋 Scenario: {args.scenario}")
    console.print(f"⏱️ Duration: {args.duration}s")
    console.print(f"👥 Concurrent Users: {args.concurrent_users}")
    console.print(f"📊 Target RPS: {args.rps}")
    console.print(f"🌐 Base URL: {args.base_url}")
    console.print(f"👤 Tenant ID: {args.tenant_id}")
    
    # Run the test
    result = run_load_test(
        scenario=args.scenario,
        duration=args.duration,
        concurrent_users=args.concurrent_users,
        requests_per_second=args.rps,
        base_url=args.base_url,
        tenant_id=args.tenant_id,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    if result.error_rate > 10:  # More than 10% errors
        console.print("\n[red]❌ Test failed: High error rate[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]✅ Test completed successfully[/green]")
        sys.exit(0)

if __name__ == "__main__":
    main()
