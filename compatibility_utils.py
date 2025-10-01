# compatibility_utils.py
import torch
import datetime

def get_pytorch_version():
    """Detect PyTorch version for compatibility"""
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    return major, minor

def setup_timeout():
    """Create timeout compatible with PyTorch version"""
    major, minor = get_pytorch_version()
    
    if major >= 1 and minor >= 10:
        # Use modern timedelta for newer PyTorch
        if hasattr(torch.distributed, 'timedelta'):
            return torch.distributed.timedelta(seconds=1800)
        else:
            # Fallback to datetime
            return datetime.timedelta(seconds=1800)
    else:
        # Use datetime for older versions
        return datetime.timedelta(seconds=1800)

def check_environment():
    """Check PyTorch environment and compatibility"""
    print("=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check distributed module
    if hasattr(torch.distributed, 'timedelta'):
        print("✓ torch.distributed.timedelta available")
    else:
        print("✗ torch.distributed.timedelta not available - using datetime")
    
    # Check critical attributes
    critical_attrs = ['init_process_group', 'is_initialized', 'destroy_process_group']
    for attr in critical_attrs:
        if hasattr(torch.distributed, attr):
            print(f"✓ torch.distributed.{attr} available")
        else:
            print(f"✗ torch.distributed.{attr} missing!")
    
    # Check timeout compatibility
    timeout = setup_timeout()
    print(f"✓ Timeout setup: {type(timeout).__name__}")

if __name__ == "__main__":
    check_environment()
