# Multi-GPU Training with 2× RTX 5090s

## 🚀 **Quick Start**

### **Multi-GPU Training (Recommended)**
```bash
torchrun --nproc_per_node=2 train.py --config runs/humigence/config.snapshot.json
```

### **Single GPU Training (Fallback)**
```bash
python train.py --config runs/humigence/config.snapshot.json --fallback_single_gpu
```

## 🔧 **Features**

### **Multi-GPU Support**
- ✅ **NCCL Backend**: Stable distributed training
- ✅ **2× RTX 5090s**: Full utilization of both GPUs
- ✅ **Automatic Detection**: Detects available GPUs
- ✅ **Process Synchronization**: Proper rank management

### **Environment Hardening**
- ✅ **NCCL Debug**: `NCCL_DEBUG=INFO` for troubleshooting
- ✅ **IB Disabled**: `NCCL_IB_DISABLE=1` prevents InfiniBand issues
- ✅ **P2P Disabled**: `NCCL_P2P_DISABLE=1` prevents peer-to-peer issues
- ✅ **Async Error Handling**: `NCCL_ASYNC_ERROR_HANDLING=1` for better error handling
- ✅ **Tokenizer Safety**: `TOKENIZERS_PARALLELISM=false` prevents fork warnings

### **Graceful Fallback**
- ✅ **Automatic Fallback**: Falls back to single GPU if multi-GPU fails
- ✅ **Clear Warnings**: Shows when fallback is triggered
- ✅ **No Data Loss**: Training continues seamlessly
- ✅ **Error Recovery**: Handles NCCL errors gracefully

### **Device Consistency**
- ✅ **Training**: Each process uses its local rank device (`cuda:local_rank`)
- ✅ **Evaluation**: Always uses `cuda:0` for fresh model reload
- ✅ **No Mixing**: No tensors mixed between `cuda:0` and `cuda:1`
- ✅ **Synchronization**: Proper process synchronization

## 📊 **Training Modes**

### **Multi-GPU Mode (Default)**
```
🚀 Starting multi-GPU training on 2 GPUs
✅ Distributed training initialized
   Rank: 0/1
   Local Rank: 0
   World Size: 2
   Device: cuda:0

🚀 Starting multi-GPU training on 2 GPUs
✅ Distributed training initialized
   Rank: 1/1
   Local Rank: 1
   World Size: 2
   Device: cuda:1
```

### **Single GPU Fallback**
```
⚠️ NCCL multi-GPU failed, falling back to single GPU training on cuda:0.
🚀 Starting single GPU training on cuda:0
✅ Single GPU Training: cuda:0
   Device: cuda:0
```

## 🛠️ **Configuration**

### **Multi-GPU Configuration**
The launcher automatically sets:
```python
config = {
    "distributed": True,
    "rank": 0,  # or 1
    "world_size": 2,
    "local_rank": 0,  # or 1
    "device": "cuda:0",  # or "cuda:1"
    "per_device_train_batch_size": 4,  # Per GPU
    "per_device_eval_batch_size": 8,   # Per GPU
}
```

### **Single GPU Configuration**
```python
config = {
    "distributed": False,
    "rank": 0,
    "world_size": 1,
    "local_rank": 0,
    "device": "cuda:0",
    "per_device_train_batch_size": 8,  # Doubled for single GPU
    "per_device_eval_batch_size": 16,  # Doubled for single GPU
}
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **NCCL Initialization Failed**
   ```
   ❌ Distributed training initialization failed: NCCL Error
   ⚠️ NCCL multi-GPU failed, falling back to single GPU training on cuda:0.
   ```
   **Solution**: This is expected behavior. The launcher will automatically fall back to single GPU.

2. **CUDA Out of Memory**
   ```
   ❌ CUDA out of memory
   ```
   **Solution**: Reduce `per_device_train_batch_size` in your config.

3. **Device Mismatch**
   ```
   ❌ Expected all tensors to be on the same device
   ```
   **Solution**: This should not happen with the new launcher. If it does, check that evaluation is using fresh model reload.

### **Debug Mode**
Set environment variables for debugging:
```bash
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=2 train.py --config runs/humigence/config.snapshot.json
```

## 📈 **Performance**

### **Multi-GPU Benefits**
- **2× Training Speed**: Approximately 2x faster training
- **Larger Batch Sizes**: Can use larger effective batch sizes
- **Better Convergence**: Often better model performance
- **Memory Efficiency**: Distributes memory across GPUs

### **Single GPU Fallback**
- **Reliable**: Always works if multi-GPU fails
- **Simpler**: Easier to debug issues
- **Compatible**: Works with any setup

## 🎯 **Best Practices**

1. **Always use the launcher**: Don't run training directly
2. **Check GPU availability**: Ensure both GPUs are visible
3. **Monitor memory usage**: Watch for OOM errors
4. **Use appropriate batch sizes**: Start small and increase
5. **Check logs**: Look for NCCL warnings or errors

## 🚨 **Important Notes**

- **Evaluation always uses cuda:0**: Fresh model reload ensures device consistency
- **Training uses local rank devices**: Each process uses its assigned GPU
- **No tensor mixing**: Tensors never cross between cuda:0 and cuda:1
- **Automatic fallback**: If multi-GPU fails, single GPU training continues
- **Process synchronization**: All processes are properly synchronized

## 🎉 **Summary**

The new training launcher provides:
- **Robust multi-GPU training** with NCCL
- **Graceful fallback** to single GPU
- **Device consistency** throughout training and evaluation
- **Professional logging** and error handling
- **Fool-proof operation** with automatic error recovery

No more `cuda:0 vs cuda:1` mismatches, no deadlocks, no NCCL crashes without fallback! 🚀
