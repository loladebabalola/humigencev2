# utils/dataset_loader.py

import os
from pathlib import Path
from typing import List, Tuple, Union

def list_local_datasets(base_dir: str = "~/humigence_data") -> List[Tuple[str, str, Union[int, str]]]:
    """
    List all local datasets in the specified directory.
    
    Args:
        base_dir: Base directory to search for datasets (default: ~/humigence_data)
        
    Returns:
        List of tuples containing (name, path, count) for each dataset
        where count is the number of lines in the JSONL file or "?" if error
    """
    base = os.path.expanduser(base_dir)
    datasets = []
    
    if not os.path.exists(base):
        return datasets
    
    try:
        for f in os.listdir(base):
            if f.endswith(".jsonl"):
                path = os.path.join(base, f)
                try:
                    # Count lines in the file
                    with open(path, "r", encoding='utf-8') as infile:
                        count = sum(1 for _ in infile)
                except Exception:
                    count = "?"
                
                # Extract name without extension
                name = os.path.splitext(f)[0]
                datasets.append((name, path, count))
    except Exception:
        # If there's any error accessing the directory, return empty list
        pass
    
    return datasets
