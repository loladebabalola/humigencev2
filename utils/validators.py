import os
import json

def detect_datasets(base_path="~/humigence_data"):
    base_path = os.path.expanduser(base_path)
    choices = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r") as f:
                        if file.endswith(".jsonl"):
                            count = sum(1 for _ in f)
                        else:
                            data = json.load(f)
                            count = len(data)
                    display_name = f"{file} ({count} samples)"
                    choices.append((display_name, full_path))
                except Exception:
                    continue
    return choices