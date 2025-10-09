# Humigence SSH Client Access Guide

## 🚀 Complete SSH-Based Authentication System

This guide shows how to set up and use the streamlined SSH-based authentication system where clients can directly access the AI model through device credentials.

## 📋 System Overview

- **Admin** configures device credentials (device name + API key)
- **Client** SSH into server and runs `humigence-client`
- **Client** enters device name and API key (password)
- **Client** gets direct access to the AI model

## 🔧 Setup Instructions

### Step 1: Setup SSH Alias (One-time setup)

```bash
cd /home/joshua/humigencev2
./setup_ssh_alias.sh
source ~/.bashrc  # or ~/.zshrc
```

### Step 2: Start the Server

```bash
humigence
# Select: 3. Multi-Tenant Inference
# Select: 1. Start/Stop Services
# Wait for "All services started successfully!"
```

### Step 3: Create Device Credentials (Admin)

```bash
python3 admin_device_manager.py
# Select: 1. Create New Device
# Enter: Device name (e.g., "iPhone-13", "MacBook-Pro")
# Enter: Quota percentage (e.g., 25%)
# Enter: Priority (e.g., normal)
# Copy the generated credentials
```

## 👥 Usage Instructions

### For Admins

#### Create Device Credentials
```bash
python3 admin_device_manager.py
```

**Options:**
- **1. Create New Device** - Add new client device
- **2. List All Devices** - View all registered devices
- **3. Update Device** - Modify device settings
- **4. Remove Device** - Delete device access
- **5. Device Statistics** - View usage stats
- **6. Generate Client Credentials** - Get login info
- **7. Test Device Connection** - Verify device works
- **8. Export Device List** - Backup device data

#### Example: Create iPhone Device
```bash
python3 admin_device_manager.py
# Select: 1. Create New Device
# Device Name: iPhone-13
# Quota: 25
# Priority: normal
# Result: Device created with API key
```

### For Clients

#### SSH and Login
```bash
# SSH into the server
ssh user@10.0.0.98

# Run the client login
humigence-client

# Enter credentials when prompted:
# Device Name: iPhone-13
# API Key (Password): [your_api_key]
```

#### Client Interface Options
After login, clients get access to:

- **1. Chat with AI** - Interactive chat session
- **2. Quick Questions** - Pre-defined question templates
- **3. Check Server Status** - View server and device status
- **4. Device Information** - View device details
- **5. Logout** - End session

## 📱 Example Workflows

### Admin Workflow: Setting Up a New Client

1. **Start Server**
   ```bash
   humigence
   # Select: 3. Multi-Tenant Inference
   # Select: 1. Start/Stop Services
   ```

2. **Create Device**
   ```bash
   python3 admin_device_manager.py
   # Select: 1. Create New Device
   # Device Name: John-iPhone
   # Quota: 30%
   # Priority: normal
   ```

3. **Provide Credentials to Client**
   ```
   Device Name: John-iPhone
   API Key: abc123def456...
   ```

4. **Client Tests Access**
   ```bash
   ssh user@10.0.0.98
   humigence-client
   # Enter: John-iPhone
   # Enter: abc123def456...
   # Success! Start chatting with AI
   ```

### Client Workflow: Daily Usage

1. **Connect to Server**
   ```bash
   ssh user@10.0.0.98
   ```

2. **Login to AI**
   ```bash
   humigence-client
   # Enter device name and API key
   ```

3. **Use AI Features**
   - Chat with AI
   - Ask quick questions
   - Check status
   - View device info

4. **Logout When Done**
   ```bash
   # Select: 5. Logout
   ```

## 🔐 Security Features

- **Unique API Keys** - Each device gets a secure, unique API key
- **Device-Specific Access** - Keys are tied to specific device names
- **Quota Management** - Prevent any device from overwhelming the server
- **Priority System** - High-priority devices get faster responses
- **Session Management** - Track device usage and last access times

## 🛠️ Troubleshooting

### Client Can't Connect
```bash
# Check if server is running
curl http://10.0.0.98:8000/health

# Check device credentials
python3 admin_device_manager.py
# Select: 7. Test Device Connection
```

### Invalid Credentials
```bash
# Verify device exists
python3 admin_device_manager.py
# Select: 2. List All Devices

# Regenerate credentials
python3 admin_device_manager.py
# Select: 6. Generate Client Credentials
```

### Server Not Responding
```bash
# Restart server
humigence
# Select: 3. Multi-Tenant Inference
# Select: 1. Start/Stop Services
```

## 📊 Device Management

### View All Devices
```bash
python3 admin_device_manager.py
# Select: 2. List All Devices
```

### Update Device Settings
```bash
python3 admin_device_manager.py
# Select: 3. Update Device
# Select device and modify quota/priority
```

### Remove Device Access
```bash
python3 admin_device_manager.py
# Select: 4. Remove Device
# Select device to remove
```

### Export Device List
```bash
python3 admin_device_manager.py
# Select: 8. Export Device List
# Saves to JSON file for backup
```

## 🎯 Best Practices

### For Admins
1. **Use descriptive device names**: `John-iPhone`, `Sarah-MacBook`, `Office-iPad`
2. **Set appropriate quotas**: 20-30% for mobile, 50%+ for workstations
3. **Use high priority for important devices**: Work laptops, production servers
4. **Regular monitoring**: Check device statistics and usage
5. **Backup device list**: Export regularly for disaster recovery

### For Clients
1. **Keep credentials secure**: Don't share API keys
2. **Logout when done**: End sessions properly
3. **Report issues**: Contact admin if problems occur
4. **Respect quotas**: Don't abuse the system

## 🔄 Maintenance

### Regular Tasks
- Monitor device usage statistics
- Update device quotas as needed
- Remove inactive devices
- Backup device configuration
- Test device connections

### Server Maintenance
- Restart server if needed
- Monitor server health
- Update device endpoints if IP changes
- Check firewall settings

## 📋 Quick Reference

### Admin Commands
```bash
# Start server
humigence

# Manage devices
python3 admin_device_manager.py

# Setup SSH alias (one-time)
./setup_ssh_alias.sh
```

### Client Commands
```bash
# SSH and login
ssh user@10.0.0.98
humigence-client

# Direct API access (alternative)
curl -X POST http://10.0.0.98:8000/[device_id]/v1/chat/completions \
  -H "Authorization: Bearer [api_key]" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b-F16", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 🎉 Benefits

- **Streamlined Access** - Simple SSH + command interface
- **Secure Authentication** - Device-specific API keys
- **Easy Management** - Admin interface for all devices
- **User-Friendly** - Interactive chat interface
- **Scalable** - Support unlimited devices
- **Monitored** - Track usage and statistics

---

**Ready to deploy! Your SSH-based AI access system is complete! 🤖🔐**

