#!/usr/bin/env python3
"""
Humigence Client for Demo-Device
Device ID: Demo-Device_7dfaee07
Generated: 2025-10-03 21:46:49
"""

import requests
import json
import sys
import argparse
from typing import Optional

class HumigenceClient:
    def __init__(self, device_id: str = "Demo-Device_7dfaee07", api_key: str = "V84nkzpQu7U4zWukI1kA2tQqnC3S3JnzheuQkPFaNdw"):
        self.device_id = device_id
        self.api_key = api_key
        self.base_url = "http://localhost:8000/Demo-Device_7dfaee07"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Tenant-ID": device_id
        }
    
    def chat(self, message: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Send a chat message to the Humigence server"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-oss-20b-F16",
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0].get('message', {}).get('content', '')
            return "No response received"
            
        except Exception as e:
            return f"Error: {e}"
    
    def status(self) -> dict:
        """Check server status"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            return {"status": "online", "response": response.json()}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Humigence Client for Demo-Device")
    parser.add_argument("message", nargs="?", help="Message to send to the AI")
    parser.add_argument("--max-tokens", "-t", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--status", action="store_true", help="Check server status")
    
    args = parser.parse_args()
    
    client = HumigenceClient()
    
    if args.status:
        status = client.status()
        print(json.dumps(status, indent=2))
    elif args.message:
        response = client.chat(args.message, args.max_tokens, args.temperature)
        print(response)
    else:
        # Interactive mode
        print(f"🤖 Humigence Client for Demo-Device")
        print(f"Device ID: Demo-Device_7dfaee07")
        print(f"Endpoint: http://localhost:8000/Demo-Device_7dfaee07")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                message = input("You: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                if message:
                    response = client.chat(message, args.max_tokens, args.temperature)
                    print(f"AI: {response}\n")
            except KeyboardInterrupt:
                break
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
