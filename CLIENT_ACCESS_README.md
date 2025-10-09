# Humigence Client Access System

## Overview

The Humigence Client Access System allows you to easily register devices and provide secure access to your AI inference server. Each device gets its own API key, endpoint, and quota management.

## 🚀 Quick Start

### 1. Start the Server
```bash
humigence
# Select option 3 (Multi-Tenant Inference)
# Wait for "All services started successfully!"
```

### 2. Register Your Device
```bash
python3 setup_client_access.py
# Select option 1 (Register New Device)
# Follow the prompts
```

### 3. Use Your Device
```bash
# Copy the generated client script to your device
python3 humigence_client_[device_id].py "Hello, AI!"

# Or use direct API calls
curl -X POST http://localhost:8000/[device_id]/v1/chat/completions \
  -H "Authorization: Bearer [api_key]" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b-F16", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 📱 Device Management

### Register a New Device
```bash
python3 setup_client_access.py
```

Options:
- **Device Name**: Custom name or auto-detect
- **Quota Percentage**: 0-100% of server capacity
- **Priority**: low/normal/high (affects response speed)

### List All Devices
```bash
python3 setup_client_access.py
# Select option 2 (List Devices)
```

### Manage Existing Device
```bash
python3 setup_client_access.py
# Select option 3 (Manage Device)
```

Options:
- Generate new client script
- Show credentials
- Test connection
- Remove device

## 🔐 Security Features

- **Unique API Keys**: Each device gets a secure, unique API key
- **Device-Specific Endpoints**: Each device has its own endpoint
- **Quota Management**: Control resource usage per device
- **Priority System**: High-priority devices get faster responses
- **Rate Limiting**: Built-in protection against abuse

## 📊 Device Information

Each registered device includes:
- **Device ID**: Unique identifier
- **Device Name**: Human-readable name
- **Device Type**: phone, tablet, laptop, desktop, server
- **OS Information**: Operating system and version
- **Hardware ID**: Unique hardware fingerprint
- **Network Info**: Hostname and IP address
- **Capabilities**: What the device can do

## 🌐 API Endpoints

### Device-Specific Endpoint
```
POST /{device_id}/v1/chat/completions
```

Headers:
```
Authorization: Bearer {api_key}
Content-Type: application/json
X-Tenant-ID: {device_id}
```

### Health Check
```
GET /health
```

## 📄 Client Scripts

Each device gets a personalized client script with:
- Pre-configured credentials
- Easy-to-use interface
- Interactive mode
- Status checking
- Error handling

### Client Script Usage
```bash
# Send a message
python3 humigence_client_[device_id].py "Hello, how are you?"

# Check server status
python3 humigence_client_[device_id].py --status

# Test connection
python3 humigence_client_[device_id].py --test

# Interactive mode
python3 humigence_client_[device_id].py
```

## 🔧 Configuration

### Device Quotas
- **Default**: 25% of server capacity
- **Range**: 0-100%
- **Purpose**: Prevents any single device from overwhelming the server

### Priority Levels
- **Low**: Standard processing
- **Normal**: Balanced processing (default)
- **High**: Priority processing for faster responses

### Rate Limiting
- **Requests per second**: Configurable per device
- **Daily limits**: Prevents abuse
- **Context size limits**: Prevents oversized requests

## 📱 Multi-Device Setup

### Step 1: Register Each Device
Run the setup script on each device:
```bash
python3 setup_client_access.py
```

### Step 2: Distribute Client Scripts
Each device gets its own client script with embedded credentials.

### Step 3: Test Connectivity
Use the built-in test functions to verify each device can connect.

## 🛠️ Troubleshooting

### Server Not Running
```
❌ Humigence server is not running
💡 Please start the server first using: humigence
```

**Solution**: Start the server with option 3 (Multi-Tenant Inference)

### Device Not Found
```
❌ Device 'device_id' not found!
```

**Solution**: Check device ID or re-register the device

### Connection Failed
```
❌ Connection test failed!
```

**Solutions**:
1. Check if server is running
2. Verify API key is correct
3. Check network connectivity
4. Ensure device is registered

### Rate Limit Exceeded
```
❌ Request validation failed: Rate limit exceeded
```

**Solution**: Wait a moment before making another request

## 📋 Example Workflows

### Register iPhone
```bash
python3 setup_client_access.py
# Device name: iPhone
# Quota: 20%
# Priority: normal
```

### Register Laptop
```bash
python3 setup_client_access.py
# Device name: MacBook-Pro
# Quota: 50%
# Priority: high
```

### Register Server
```bash
python3 setup_client_access.py
# Device name: Production-Server
# Quota: 30%
# Priority: normal
```

## 🔄 Device Lifecycle

1. **Registration**: Device is registered and gets credentials
2. **Active**: Device can make requests within quota
3. **Monitoring**: Usage is tracked and logged
4. **Management**: Quota and priority can be adjusted
5. **Removal**: Device can be removed when no longer needed

## 📈 Monitoring

### Device Statistics
- Total requests made
- Total tokens generated
- Current requests per second
- Error count
- Last seen timestamp

### System Overview
- Total devices registered
- Active devices
- Total quota allocated
- Server health status

## 🚀 Advanced Features

### Custom Client Scripts
Generate custom client scripts with specific configurations:
```bash
python3 -c "
from inference.device_manager import DeviceManager
dm = DeviceManager()
dm.generate_client_script('device_id', 'custom_client.py')
"
```

### Programmatic Access
```python
from inference.device_manager import DeviceManager

dm = DeviceManager()
device = dm.get_device('device_id')
print(f"API Key: {device.api_key}")
print(f"Endpoint: {device.endpoint}")
```

### Bulk Device Management
```python
# Register multiple devices
devices = [
    {'name': 'iPhone', 'quota': 20, 'priority': 'normal'},
    {'name': 'iPad', 'quota': 15, 'priority': 'low'},
    {'name': 'MacBook', 'quota': 50, 'priority': 'high'}
]

for device_info in devices:
    device_id, api_key = dm.register_device(**device_info)
    print(f"Registered {device_info['name']}: {device_id}")
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify server status
3. Test device connectivity
4. Check device registration

## 🔄 Updates

The client access system is designed to be:
- **Backward Compatible**: Existing devices continue to work
- **Forward Compatible**: New features don't break old clients
- **Self-Healing**: Automatic reconnection and error recovery
- **Scalable**: Supports unlimited devices

---

**Happy AI-ing! 🤖✨**

