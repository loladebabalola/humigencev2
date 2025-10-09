#!/usr/bin/env python3
"""
Humigence Client Template
========================

This is a template client script for accessing Humigence AI inference server.
Replace the placeholders below with your actual device credentials.

To get your credentials:
1. Run: python3 setup_client_access.py
2. Register your device
3. Copy the generated client script

Usage:
    python3 humigence_client_template.py "Hello, how are you?"
    python3 humigence_client_template.py --status
    python3 humigence_client_template.py  # Interactive mode
"""

import requests
import json
import sys
import argparse
from typing import Optional

# =============================================================================
# DEVICE CREDENTIALS - REPLACE WITH YOUR ACTUAL VALUES
# =============================================================================
DEVICE_ID = "YOUR_DEVICE_ID_HERE"
API_KEY = "YOUR_API_KEY_HERE"
BASE_URL = "http://localhost:8000"  # Change if server is on different host/port
# =============================================================================

class HumigenceClient:
    """Client for accessing Humigence AI inference server"""
    
    def __init__(self, device_id: str = DEVICE_ID, api_key: str = API_KEY, base_url: str = BASE_URL):
        self.device_id = device_id
        self.api_key = api_key
        self.base_url = base_url
        self.endpoint = f"{base_url}/{device_id}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Tenant-ID": device_id
        }
    
    def chat(self, message: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Send a chat message to the Humigence server"""
        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
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
            
        except requests.exceptions.RequestException as e:
            return f"Request error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"
    
    def status(self) -> dict:
        """Check server status"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return {"status": "online", "response": response.json()}
            else:
                return {"status": "error", "code": response.status_code, "message": response.text}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    def test_connection(self) -> bool:
        """Test if the connection to the server works"""
        try:
            # Test with a simple message
            response = self.chat("Test", max_tokens=5)
            return not response.startswith("Request error") and not response.startswith("Unexpected error")
        except:
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Humigence AI Client")
    parser.add_argument("message", nargs="?", help="Message to send to the AI")
    parser.add_argument("--max-tokens", "-t", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (0.0-2.0)")
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument("--test", action="store_true", help="Test connection")
    
    args = parser.parse_args()
    
    # Check if credentials are set
    if DEVICE_ID == "YOUR_DEVICE_ID_HERE" or API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Error: Device credentials not configured!")
        print("Please edit this script and replace the placeholder values:")
        print(f"   DEVICE_ID = \"{DEVICE_ID}\"")
        print(f"   API_KEY = \"{API_KEY}\"")
        print("\nTo get your credentials, run: python3 setup_client_access.py")
        sys.exit(1)
    
    client = HumigenceClient()
    
    if args.status:
        status = client.status()
        print(json.dumps(status, indent=2))
    
    elif args.test:
        print("🔄 Testing connection...")
        if client.test_connection():
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed!")
            print("Make sure the Humigence server is running and your credentials are correct.")
    
    elif args.message:
        print(f"🤖 Sending: {args.message}")
        response = client.chat(args.message, args.max_tokens, args.temperature)
        print(f"AI: {response}")
    
    else:
        # Interactive mode
        print(f"🤖 Humigence AI Client")
        print(f"Device ID: {DEVICE_ID}")
        print(f"Endpoint: {client.endpoint}")
        print("Type 'quit' to exit, 'status' for server status, 'test' to test connection\n")
        
        while True:
            try:
                message = input("You: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                elif message.lower() == 'status':
                    status = client.status()
                    print(f"Server Status: {json.dumps(status, indent=2)}\n")
                elif message.lower() == 'test':
                    print("🔄 Testing connection...")
                    if client.test_connection():
                        print("✅ Connection test successful!\n")
                    else:
                        print("❌ Connection test failed!\n")
                elif message:
                    response = client.chat(message, args.max_tokens, args.temperature)
                    print(f"AI: {response}\n")
            except KeyboardInterrupt:
                break
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()

