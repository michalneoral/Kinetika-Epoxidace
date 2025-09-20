# config_manager.py
import json
import os
from datetime import datetime

CONFIG_DIR = "configs"
RECENT_CONFIG = os.path.join(CONFIG_DIR, "recent_config.json")

def ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config(path=RECENT_CONFIG):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(data, path=RECENT_CONFIG):
    ensure_config_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_configs():
    ensure_config_dir()
    return sorted([
        f for f in os.listdir(CONFIG_DIR)
        if f.endswith(".json") and f != "recent_config.json"
    ])

def get_timestamped_filename():
    dt = datetime.now().strftime("%Y%m%d")
    return f"config_{dt}.json"
