
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_unsloth():
    try:
        from training.unsloth.launcher import check_gpu_availability
        gpu_info = check_gpu_availability()
        print(f"GPU Info: {gpu_info}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_unsloth()
    sys.exit(0 if success else 1)
