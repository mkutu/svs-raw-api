"""
Command-line interface for svs-raw-api.
"""

import argparse
import sys
from pathlib import Path
import logging

import numpy as np
import yaml

from svs_raw_api import SVSRaw2DNG, __version__



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger('svs-convert')


def load_raw_image(raw_path: Path, height: int, width: int) -> np.ndarray:
    """Load and reshape RAW image data."""
    try:
        raw_data = np.fromfile(raw_path, dtype=np.uint16)
        return raw_data.reshape((height, width))
    except Exception as e:
        raise ValueError(f"Failed to load RAW image {raw_path}: {e}")
    
def load_camera_tags(tags_path: Path) -> dict:
    """Load camera tags from YAML file."""
    try:
        with open(tags_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load camera tags file {tags_path}: {e}")
    
def load_color_matrix(matrix_path: Path) -> np.ndarray:
    """Load color calibration matrix from .npy file."""
    try:
        return np.load(matrix_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load color matrix {matrix_path}: {e}")
    
def get_image_dimensions(tags: dict) -> tuple:
    """Extract image dimensions from camera tags."""
    try:
        height = tags['image']['SVCamImageHeight']
        width = tags['image']['SVCamImageWidth']
        return height, width
    except KeyError as e:
        raise ValueError(f"Image dimensions not found in camera tags: {e}")

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

def process_files(input_path: Path, output_path: Path, converter: SVSRaw2DNG, camera_tags: dict):
    # Single file conversion
    logger.info(f"Converting {input_path} → {output_path}")
    try:
        height, width = get_image_dimensions(camera_tags)
        raw_image = load_raw_image(input_path, height, width)
        converter.save_dng(raw_image, output_path, camera_tags)
        logger.info(f"Successfully created {output_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)



def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Sony ARW/RAW images to DNG format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single image
  svs-convert -i image.ARW -o output.dng -m matrix.npy -t tags.yaml
  
  # Batch convert directory
  svs-convert -i input_dir/ -o output_dir/ -m matrix.npy -t tags.yaml
  
  # Custom dimensions
  svs-convert -i image.ARW -o output.dng -m matrix.npy --width 4032 --height 3024
        """
    )
    
    parser.add_argument('-i', '--config', required=True, type=Path,
                        help='Input RAW file or directory')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading configuration from {args.config}")
    cfg = load_config(args.config)

    # Load color matrix
    color_matrix_path = cfg['paths']['color_matrix']    
    logger.info(f"Loading color matrix from {color_matrix_path}")
    color_matrix = load_color_matrix(color_matrix_path)
    
    # Load camera tags
    camera_tags_path = cfg['paths']['svs_tags']
    logger.info(f"Loading camera tags from {camera_tags_path}")
    camera_tags = load_camera_tags(camera_tags_path)
    
    # Get width and height from camera tags if available
    height, width = get_image_dimensions(camera_tags)
    logger.info(f"Image dimensions from tags: {width}x{height}")
    
    # Create converter
    logger.info("Initializing converter")
    converter = SVSRaw2DNG(
        color_matrix=color_matrix
    )
    
    # Process files
    input_path = cfg['paths']['input']
    output_path = cfg['paths']['output']
    
    if input_path.is_file():
        # Single file conversion
        logger.info(f"Converting {input_path} → {output_path}")
        try:
            raw_image = load_raw_image(input_path, height, width)
            converter.save_dng(raw_image, output_path, camera_tags)
            logger.info(f"Successfully created {output_path}")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Batch conversion
        output_path.mkdir(parents=True, exist_ok=True)
        
        raw_files = list(input_path.glob("*.ARW")) + list(input_path.glob("*.RAW"))
        logger.info(f"Found {len(raw_files)} RAW files in {input_path}")
        
        success_count = 0
        fail_count = 0
        
        for raw_file in raw_files:
            output_file = output_path / f"{raw_file.stem}.dng"
            logger.info(f"Converting {raw_file.name} → {output_file.name}")
            
            try:
                height, width = get_image_dimensions(camera_tags)
                raw_image = load_raw_image(raw_file, height, width)
                converter.save_dng(raw_image, output_file, camera_tags)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to convert {raw_file.name}: {e}")
                fail_count += 1
        
        logger.info(f"Batch conversion complete: {success_count} success, {fail_count} failed")
        
        if fail_count > 0:
            sys.exit(1)
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
