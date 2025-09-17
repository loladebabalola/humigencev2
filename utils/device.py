import platform
import psutil
import torch
import subprocess

def get_system_info():
    info = {
        "Platform": platform.system(),
        "Python Version": platform.python_version(),
        "Torch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda,
        "RAM": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB",
        "CPUs": psutil.cpu_count(logical=True),
    }

    if torch.cuda.is_available():
        info["GPU Count"] = torch.cuda.device_count()
        info["GPUs"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory": f"{round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)} GB"
            } for i in range(torch.cuda.device_count())
        ]
    else:
        info["GPU Count"] = 0
        info["GPUs"] = []

    return info