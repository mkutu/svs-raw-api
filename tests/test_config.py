#!/usr/bin/env python3
"""
Test config loading and path expansion
Usage: python scripts/test_config.py
"""
from pathlib import Path
import sys

# Add repo to path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))

from svs_raw_api.cli import load_config

config_path = repo_dir / "conf" / "scinet.yaml"
print(f"Loading config from: {config_path}")
print(f"Config exists: {config_path.exists()}")
print()

try:
    config = load_config(config_path)
    print("✓ Config loaded successfully!")
    print()
    
    print("Paths:")
    for key, value in config['paths'].items():
        exists = Path(value).exists() if not key.endswith('_base') else None
        status = "✓" if exists else "✗" if exists is False else "-"
        print(f"  {status} {key:20s}: {value}")
    
    print()
    print("Processing settings:")
    for key, value in config['processing'].items():
        print(f"  {key:20s}: {value}")
    
    print()
    print("SLURM settings:")
    for key, value in config['slurm'].items():
        print(f"  {key:20s}: {value}")
    
except Exception as e:
    print(f"✗ Error loading config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)