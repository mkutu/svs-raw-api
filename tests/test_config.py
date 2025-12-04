#!/usr/bin/env python3
"""
Test configuration loading and path validation
"""

import sys
from pathlib import Path
import yaml

def test_config():
    """Test configuration loading."""
    config_path = Path(__file__).parent.parent / "config" / "scinet.yaml"
    
    print(f"Testing configuration: {config_path}")
    print(f"Config exists: {config_path.exists()}\n")
    
    if not config_path.exists():
        print("❌ Config file not found!")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print("✅ Config loaded successfully!\n")
        
        # Check paths
        print("Paths:")
        for key, value in config.get('paths', {}).items():
            path = Path(value)
            # Don't check existence for paths that might not exist yet
            if '_base' in key or key == 'repo_root':
                print(f"  ⚬ {key:20s}: {value}")
            else:
                exists = path.exists()
                status = "✅" if exists else "❌"
                print(f"  {status} {key:20s}: {value}")
        
        # Check processing settings
        print("\nProcessing settings:")
        for key, value in config.get('processing', {}).items():
            print(f"  ⚬ {key:20s}: {value}")
        
        # Check SLURM settings
        print("\nSLURM settings:")
        for key, value in config.get('slurm', {}).items():
            print(f"  ⚬ {key:20s}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_config()
    sys.exit(0 if success else 1)
