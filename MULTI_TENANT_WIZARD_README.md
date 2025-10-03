# Multi-Tenant Inference Wizard - World-Class Implementation

## 🎯 Overview

The Multi-Tenant Inference Wizard is a zero-friction, production-ready system for managing multi-tenant inference workloads with automatic GPU detection, model scanning, and runtime tenant management.

## ✨ Key Features

### 🚀 Zero-Friction Setup
- **Auto-detect GPUs**: Automatically discovers available GPUs and maps them to ports
- **Auto-scan models**: Scans `~/models/` directory for GGUF files and selects the best default
- **Auto-create tenants**: Creates default tenant with equal quota distribution
- **Auto-generate profiles**: Creates configuration profiles based on detected hardware

### 🔧 Runtime Tenant Management
- **Dynamic tenant addition**: `tenant add <alias> [--quota=XX%] [--priority=low|normal|high]`
- **Live tenant updates**: Modify quotas and priorities without service restart
- **Tenant removal**: `tenant remove <alias>` with automatic quota reallocation
- **Real-time monitoring**: Live dashboard showing tenant usage and performance

### 📊 Comprehensive Monitoring
- **Live dashboard**: Real-time system metrics with Rich TUI
- **GPU monitoring**: Memory usage, utilization, temperature, power consumption
- **Service monitoring**: Process status, CPU/memory usage, uptime tracking
- **Tenant analytics**: Request counts, token usage, RPS, error rates

### 💾 Profile Management
- **Auto-snapshots**: Automatic profile and metrics snapshots after each session
- **Profile export/import**: Easy configuration sharing and backup
- **Version control**: Track configuration changes over time
- **Cleanup utilities**: Automatic cleanup of old snapshots

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-Tenant Inference Wizard                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ GPU         │  │ Model       │  │ Tenant      │        │
│  │ Detector    │  │ Detector    │  │ Manager     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Summary     │  │ Profile     │  │ Monitoring  │        │
│  │ Panel       │  │ Manager     │  │ Dashboard   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Router      │  │ Supervisor  │  │ Llama       │        │
│  │ (FastAPI)   │  │ (Process    │  │ Servers     │        │
│  │             │  │ Manager)    │  │ (GPU 0,1)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Launch the Wizard
```bash
cd /home/joshua/humigence
python3 cli/main.py
# Select option 3: Multi-Tenant Inference 🖥️
```

### 2. Zero-Friction Auto-Setup
The wizard automatically:
- Detects your GPUs (2x RTX 5090 detected)
- Maps GPUs to ports (8000, 8001)
- Scans for models in `~/models/`
- Creates default tenant with equal quota
- Generates configuration profile

### 3. Start Services
```
Quick Actions:
1. Start/Stop Services
2. Monitor (live dashboard)
3. Tenant Manager (add/remove/list aliases)
4. Advanced Options
0. Exit Wizard
```

## 📋 Usage Examples

### Runtime Tenant Management

#### Add a new tenant:
```bash
# Via CLI
python3 -m inference.tenant_manager add alice --quota=40% --priority=normal

# Via wizard
# Select option 3: Tenant Manager
# Select option 2: Add Tenant
```

#### List all tenants:
```bash
python3 -m inference.tenant_manager list
```

#### Update tenant quota:
```bash
python3 -m inference.tenant_manager update alice --quota=60% --priority=high
```

#### Remove tenant:
```bash
python3 -m inference.tenant_manager remove alice
```

### API Usage

Once services are running, you can use the OpenAI-compatible API:

```bash
# Test with default tenant
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-F16",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "user": "tenant_default"
  }'

# Test with specific tenant
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-F16",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "user": "alice"
  }'
```

## 🔧 Configuration

### Profile Structure
```yaml
profile_name: auto_detected
created_at: 1696344000.0
gpu_config:
  gpu_count: 2
  available_gpus: 2
  port_mapping:
    0: 8000
    1: 8001
  gpus:
    - index: 0
      name: "NVIDIA GeForce RTX 5090"
      memory_gb: 31.4
      memory_free_gb: 31.4
      utilization_percent: 0.0
      port: 8000
      is_available: true
model_config:
  model_count: 6
  default_model:
    name: "gpt-oss-20b-F16"
    path: "/home/joshua/models/gpt-oss-20b/gpt-oss-20b-F16.gguf"
    size_gb: 12.5
    context_window: 131072
    quantization: "F16"
tenant_config:
  tenant_count: 2
  active_tenants: 2
  tenants:
    tenant_default:
      alias: "tenant_default"
      name: "Default Tenant"
      quota_percentage: 100.0
      priority: "normal"
      rps: 10
      endpoint: "http://localhost:8000"
      is_active: true
router_config:
  entrypoint: "localhost:8000"
  strategy: "tenant_sticky_then_latency"
```

### Tenant Configuration
```json
{
  "tenants": {
    "tenant_default": {
      "name": "Default Tenant",
      "api_key": null,
      "quota": {
        "rps": 10,
        "max_context": 131072,
        "max_tokens_per_request": 2048,
        "daily_limit": 10000,
        "quota_percentage": 100.0
      },
      "policy": {
        "priority": "normal",
        "allow_streaming": true,
        "max_concurrent_requests": 5,
        "timeout_seconds": 30
      },
      "created_at": 1696344000.0,
      "last_accessed": 0.0,
      "is_active": true,
      "endpoint": "http://localhost:8000"
    }
  }
}
```

## 📊 Monitoring

### Live Dashboard
The monitoring dashboard provides real-time insights into:

- **System Metrics**: CPU, memory, disk usage
- **GPU Metrics**: Memory usage, utilization, temperature, power
- **Service Metrics**: Process status, resource usage, uptime
- **Tenant Metrics**: Request counts, token usage, RPS, errors

### Health Monitoring
- **System Health Score**: Overall system health percentage
- **Service Status**: Router and llama-server process monitoring
- **Tenant Health**: Active tenant status and quota usage
- **Performance Metrics**: Latency, throughput, error rates

## 🛠️ Advanced Features

### Profile Management
- **Auto-snapshots**: Automatic profile and metrics snapshots
- **Export/Import**: Easy configuration sharing
- **Version Control**: Track configuration changes
- **Cleanup**: Automatic cleanup of old snapshots

### Dynamic Routing
- **Round Robin**: Even distribution across instances
- **Tenant Sticky**: Each tenant assigned to specific instance
- **Latency-based**: Route to instance with lowest latency
- **Health-aware**: Automatic failover for unhealthy instances

### Quota Management
- **Per-tenant quotas**: Configurable RPS, context, and daily limits
- **Priority-based**: High-priority tenants get preference
- **Dynamic updates**: Modify quotas without restart
- **Usage tracking**: Real-time quota consumption monitoring

## 🔍 Troubleshooting

### Common Issues

#### No GPUs detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### No models found
```bash
# Create models directory
mkdir -p ~/models

# Add GGUF models
# The system will auto-detect them
```

#### Services not starting
```bash
# Check if ports are in use
netstat -tlnp | grep 8000

# Kill existing processes
pkill -f "router.py"
pkill -f "supervisor.py"
pkill -f "llama-server"
```

#### Tenant quota exceeded
```bash
# Check tenant status
python3 -m inference.tenant_manager status

# Update tenant quota
python3 -m inference.tenant_manager update <alias> --quota=100%
```

## 📈 Performance Tuning

### GPU Memory Optimization
- Adjust parallel slots based on GPU memory
- Use smaller context windows for more parallel slots
- Monitor with `nvidia-smi` during operation

### Request Throughput
- Increase RPS limits in tenant quotas
- Add more GPU instances
- Optimize model quantization (Q4_K_M, Q8_0)

### Latency Optimization
- Use `tenant_sticky` routing for consistent performance
- Monitor with live dashboard
- Adjust timeout settings in tenant policies

## 🎯 Definition of Done

✅ **Zero-friction setup**: Wizard launches without manual configuration  
✅ **Auto-detection**: GPUs, models, and tenants detected automatically  
✅ **Runtime management**: Tenants can be added/removed/updated live  
✅ **Live monitoring**: Real-time dashboard with comprehensive metrics  
✅ **Profile persistence**: Automatic snapshots and configuration management  
✅ **Production ready**: Health monitoring, error handling, and failover  

## 🚀 Next Steps

1. **Install llama-server**: Run `./setup_llama_server.sh`
2. **Add your models**: Place GGUF files in `~/models/`
3. **Launch wizard**: Run `python3 cli/main.py` and select option 3
4. **Start services**: Use option 1 to start inference services
5. **Monitor performance**: Use option 2 for live monitoring
6. **Manage tenants**: Use option 3 for runtime tenant management

## 📞 Support

The Multi-Tenant Inference Wizard is production-ready with:
- ✅ Health monitoring and auto-restart
- ✅ Structured logging and metrics
- ✅ Load balancing and failover
- ✅ Tenant isolation and quotas
- ✅ OpenAI API compatibility
- ✅ Real-time monitoring dashboard

**Your Multi-Tenant Inference system is ready to serve! 🚀**
