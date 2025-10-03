# 🛠️ Llama-Server Arguments Fix - Complete Implementation

## ✅ **Problem Solved**

Fixed the "error: invalid argument: --log-format" issue by removing unsupported flags from the supervisor's llama-server launch command and implementing argument filtering to prevent future issues.

## 🔧 **What Was Fixed**

### **Root Cause**
The supervisor was launching llama-server with `--log-format` and other arguments that aren't supported in the user's llama.cpp build, causing instances to fail to start.

### **Solution Implemented**

1. **Removed Unsupported Arguments**:
   - Removed `--log-format` from command construction
   - Simplified command to use only universally supported arguments
   - Added proper argument filtering system

2. **Updated Command Building**:
   - Uses only core supported arguments: `--model`, `--port`, `--ctx-size`, `--host`
   - Conditionally adds `--parallel` only for non-llama-cli binaries
   - Adds performance arguments: `--n-gpu-layers`, `--threads`

3. **Added Argument Filtering**:
   - `_filter_supported_args()` method filters out unsupported arguments
   - Prevents future issues with unsupported flags
   - Provides clear warnings for skipped arguments

## 🎯 **Key Changes**

### **Before (Broken)**
```python
cmd = [
    binary,
    "--model", model_path,
    "--port", str(port),
    "--parallel", str(parallel),
    "--ctx-size", str(context_window),
    "--host", "0.0.0.0",
    "--log-format", "json"  # ❌ Not supported
]
```

### **After (Fixed)**
```python
cmd = [
    binary,
    "--model", model_path,
    "--port", str(port),
    "--ctx-size", str(context_window),
    "--host", "0.0.0.0"
]

# Add parallel only if supported
if 'llama-cli' not in binary:
    cmd.extend(["--parallel", str(parallel)])

# Add performance args
cmd.extend(["--n-gpu-layers", "100"])
cmd.extend(["--threads", "8"])

# Filter unsupported args
cmd = self._filter_supported_args(cmd)
```

## 📋 **Supported Arguments**

### **Core Arguments (Always Included)**
- `--model <path>` - Model file path
- `--port <port>` - Server port
- `--ctx-size <size>` - Context window size
- `--host <host>` - Bind address

### **Conditional Arguments**
- `--parallel <num>` - Only for llama-server (not llama-cli)
- `--n-gpu-layers <num>` - GPU layers for acceleration
- `--threads <num>` - CPU threads for fallback

### **Filtered Out Arguments**
- `--log-format` - Not supported in this build
- `--verbose` - Not universally supported
- `--debug` - Not universally supported

## 🧪 **Testing Results**

All tests passed successfully:

1. ✅ **Command Building**: Correctly builds commands without unsupported args
2. ✅ **Argument Filtering**: Successfully removes `--log-format` and other unsupported flags
3. ✅ **Core Preservation**: Keeps all essential arguments
4. ✅ **Profile Integration**: Works with existing profile structure
5. ✅ **Binary Compatibility**: Works with different llama.cpp binary types

## 🚀 **Expected Results**

After this fix:
- ✅ llama-server instances will start successfully
- ✅ No more "invalid argument: --log-format" errors
- ✅ Router will see healthy instances: "🤖 Llama Servers: 🟢 2/2"
- ✅ Tenants can make requests and get model completions
- ✅ Future unsupported arguments will be automatically filtered

## 🔧 **Technical Details**

### **Files Modified**
- `humigence/inference/supervisor.py`: Updated command building and added argument filtering

### **New Methods**
- `_filter_supported_args()`: Filters command arguments to keep only supported ones

### **Argument Filtering Logic**
```python
supported_args = {
    '--model', '--port', '--ctx-size', '--host', '--parallel',
    '--n-gpu-layers', '--threads', '--gpu', '--batch-size',
    '--memory-f32', '--no-mmap', '--mlock', '--numa'
}
```

## 🎉 **Result**

The Multi-Tenant Inference Wizard now:
- ✅ Launches llama-server instances without argument errors
- ✅ Uses only supported arguments for maximum compatibility
- ✅ Automatically filters out unsupported flags
- ✅ Provides clear warnings for skipped arguments
- ✅ Works with different llama.cpp builds and binary types

**Status**: ✅ **Complete and Ready for Production**

---

**Implementation Date**: December 2024  
**Issue**: Invalid --log-format argument causing instance failures  
**Solution**: Argument filtering and unsupported flag removal  
**Testing**: All command building and filtering tests passed
