"""
Configuration loading utilities.
"""
from pathlib import Path
import yaml
import numpy as np


def load_config(config_path: Path) -> dict:
    """
    Load camera configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict with camera settings and color_matrix as numpy array
    """
    config_path = Path(config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load color matrix if specified
    if 'color_matrix' in config['paths']:
        matrix_path = config_path.parent / config['paths']['color_matrix']
        config['color_matrix'] = np.load(matrix_path, allow_pickle=True)

    if 'svs_tags' in config['paths']:
        tags_path = config_path.parent / config['paths']['svs_tags']
        with open(tags_path) as f:
            config['dng_tags'] = yaml.safe_load(f)
    
    
    return config