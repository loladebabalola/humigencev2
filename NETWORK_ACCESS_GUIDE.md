# Humigence Network Access Guide

## 🌐 Accessing the Server from Other Devices

Your Humigence server is now configured for network access! Here's how to connect from other devices.

## 📍 Server Information

- **Server IP**: `10.0.0.98` (your VM's IP address)
- **Port**: `8000`
- **Base URL**: `http://10.0.0.98:8000`

## 🚀 Quick Start

### 1. Start the Server (on VM)
```bash
humigence
# Select option 3 (Multi-Tenant Inference)
# Select option 1 (Start/Stop Services)
# Wait for "All services started successfully!"
```

### 2. Register Devices (on VM)
```bash
# In the Multi-Tenant Inference menu:
# Select option 3 (Tenant Manager)
# Select option 6 (Device Management 📱)
# Select option 1 (Register New Device)
```

### 3. Access from Other Devices

#### Option A: Use Generated Client Scripts
1. Copy the generated client script to your device
2. Run: `python3 humigence_client_[device_id].py "Hello, AI!"`

#### Option B: Direct API Access
```bash
curl -X POST http://10.0.0.98:8000/[device_id]/v1/chat/completions \
  -H "Authorization: Bearer [api_key]" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b-F16", "messages": [{"role": "user", "content": "Hello!"}]}'
```

#### Option C: Python Client
```python
import requests

# Your device credentials
device_id = "your_device_id"
api_key = "your_api_key"
server_url = "http://10.0.0.98:8000"

# Send a message
response = requests.post(
    f"{server_url}/{device_id}/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Tenant-ID": device_id
    },
    json={
        "model": "gpt-oss-20b-F16",
        "messages": [{"role": "user", "content": "Hello from my device!"}],
        "max_tokens": 100
    }
)

print(response.json())
```

## 🔧 Device Registration Process

### From the VM (Server):
1. Run `humigence`
2. Select **3. Multi-Tenant Inference**
3. Select **1. Start/Stop Services** (if not already running)
4. Select **3. Tenant Manager**
5. Select **6. Device Management 📱**
6. Select **1. Register New Device**
7. Enter device details:
   - Device name: `iPhone`, `MacBook`, `iPad`, etc.
   - Quota: `25%` (or your preference)
   - Priority: `normal` (or `high` for important devices)
8. Copy the generated credentials and client script

### From Client Device:
1. Copy the client script to your device
2. Install Python 3 and requests: `pip install requests`
3. Run the client script: `python3 humigence_client_[device_id].py "Hello!"`

## 📱 Device-Specific Instructions

### iPhone/iPad
1. Install Python 3 (via App Store or other means)
2. Copy the client script to your device
3. Run: `python3 humigence_client_[device_id].py "Hello from iPhone!"`

### Android
1. Install Termux or similar terminal app
2. Install Python: `pkg install python`
3. Install requests: `pip install requests`
4. Copy and run the client script

### Windows/Mac/Linux Laptop
1. Install Python 3
2. Install requests: `pip install requests`
3. Copy and run the client script

### Other Devices
- Any device that can make HTTP requests can access the server
- Use the direct API approach with your device's credentials

## 🔐 Security Notes

- Each device gets its own unique API key
- API keys are tied to specific device endpoints
- Quota limits prevent any device from overwhelming the server
- All communication is over HTTP (consider HTTPS for production)

## 🛠️ Troubleshooting

### Connection Refused
```
curl: (7) Failed to connect to 10.0.0.98 port 8000: Connection refused
```
**Solution**: Make sure the server is running on the VM

### Device Not Found
```
❌ Device 'device_id' not found!
```
**Solution**: Make sure the device is registered on the server

### Authentication Failed
```
❌ Authentication failed
```
**Solution**: Check that you're using the correct API key

### Network Unreachable
```
❌ Network unreachable
```
**Solution**: 
1. Check that both devices are on the same network
2. Verify the server IP address
3. Check firewall settings

## 📊 Monitoring

### Check Server Status
```bash
curl http://10.0.0.98:8000/health
```

### List Registered Devices
From the VM, in the Multi-Tenant Inference menu:
- Select **3. Tenant Manager**
- Select **6. Device Management 📱**
- Select **2. List Devices**

### Test Device Connection
From the VM, in the Multi-Tenant Inference menu:
- Select **3. Tenant Manager**
- Select **6. Device Management 📱**
- Select **4. Test Device Connection**

## 🌍 Network Configuration

### Firewall (if needed)
If you have firewall issues, open port 8000:
```bash
# Ubuntu/Debian
sudo ufw allow 8000

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### Port Forwarding (if needed)
If accessing from outside your local network, set up port forwarding on your router:
- External Port: 8000
- Internal IP: 10.0.0.98
- Internal Port: 8000
- Protocol: TCP

## 📋 Example Workflows

### Register iPhone
1. On VM: Register device "iPhone" with 20% quota
2. On iPhone: Copy client script and run
3. Test: `python3 humigence_client_iPhone_abc123.py "Hello from iPhone!"`

### Register MacBook
1. On VM: Register device "MacBook-Pro" with 50% quota, high priority
2. On MacBook: Copy client script and run
3. Test: `python3 humigence_client_MacBook-Pro_def456.py "Hello from MacBook!"`

### Register Server
1. On VM: Register device "Production-Server" with 30% quota
2. On Server: Use direct API calls with credentials
3. Test: Make API calls to `http://10.0.0.98:8000/Production-Server-def456/`

## 🎯 Best Practices

1. **Use descriptive device names**: `iPhone-13`, `MacBook-Pro-2023`, `iPad-Air`
2. **Set appropriate quotas**: 20-30% for mobile devices, 50%+ for workstations
3. **Use high priority for important devices**: Work laptops, production servers
4. **Keep API keys secure**: Don't share them or commit to version control
5. **Monitor usage**: Check device statistics regularly
6. **Test connections**: Use the built-in test functions

## 🔄 Updates

The system automatically detects your network IP and uses it for all device registrations. If your IP changes, you may need to re-register devices or update the client scripts.

---

**Happy AI-ing from any device! 🤖📱💻**

