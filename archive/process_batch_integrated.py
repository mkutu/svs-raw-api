#!/usr/bin/env python3
"""
Database-integrated batch processor for svs-raw-api.

This script:
1. Queries agir-db for batches needing processing
2. Processes RAW → DNG → JPG
3. Updates database with results
4. Logs all events
"""

import argparse
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

# Import from svs-raw-api
from svs_raw_api import SVSRaw2DNG

# Import database interface (this should be added to svs-raw-api)
from svs_raw_api.db_interface import (
    get_db_connection,
    sync_batch_inventory,
    get_batches_for_processing,
    get_batch_files,
    start_batch_processing,
    complete_batch_processing,
    log_processing_event,
    get_batch_status,
)


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_camera_tags(tags_path: Path) -> Dict:
    """Load camera tags from YAML."""
    with open(tags_path) as f:
        return yaml.safe_load(f)


# ============================================================
# RAW → DNG Processing
# ============================================================

def process_raw_to_dng(
    batch_id: str,
    raw_files: List[Dict],
    config: Dict,
    camera_tags: Dict,
    color_matrix: np.ndarray,
    job_id: str = None
) -> List[Dict]:
    """
    Process RAW files to DNG format.
    
    Args:
        batch_id: Batch identifier
        raw_files: List of RAW file records from database
        config: Processing configuration
        camera_tags: Camera metadata tags
        color_matrix: Color calibration matrix
        job_id: SLURM job ID
    
    Returns:
        List of processing events
    """
    logger.info(f"Processing {len(raw_files)} RAW files for batch {batch_id}")
    
    converter = SVSRaw2DNG(color_matrix=color_matrix)
    events = []
    
    output_dir = Path(config['paths']['output_base']) / batch_id / 'dngs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    height = camera_tags['image']['SVCamImageHeight']
    width = camera_tags['image']['SVCamImageWidth']
    
    node_name = os.environ.get('SLURMD_NODENAME', 'unknown')
    
    for file_rec in raw_files:
        file_name = file_rec['file_name']
        input_path = Path(file_rec['root_path']) / file_rec['rel_path']
        output_path = output_dir / f"{Path(file_name).stem}.dng"
        
        start_time = time.time()
        
        try:
            # Load RAW image
            raw_data = np.fromfile(input_path, dtype=np.uint16)
            raw_image = raw_data.reshape((height, width))
            
            # Convert to DNG
            converter.save_dng(raw_image, str(output_path), camera_tags)
            
            processing_time = time.time() - start_time
            
            events.append({
                'batch_id': batch_id,
                'file_name': file_name,
                'pipeline_stage': 'raw_to_dng',
                'status': 'success',
                'input_path': str(input_path),
                'output_path': str(output_path),
                'processing_time_sec': processing_time,
                'error_message': None,
                'job_id': job_id,
                'node_name': node_name
            })
            
            logger.info(f"✓ {file_name} → {output_path.name} ({processing_time:.2f}s)")
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            events.append({
                'batch_id': batch_id,
                'file_name': file_name,
                'pipeline_stage': 'raw_to_dng',
                'status': 'failed',
                'input_path': str(input_path),
                'output_path': str(output_path) if output_path else None,
                'processing_time_sec': processing_time,
                'error_message': error_msg,
                'job_id': job_id,
                'node_name': node_name
            })
            
            logger.error(f"✗ {file_name}: {error_msg}")
    
    return events


# ============================================================
# DNG → JPG Processing
# ============================================================

def process_dng_to_jpg(
    batch_id: str,
    dng_files: List[Dict],
    config: Dict,
    job_id: str = None
) -> List[Dict]:
    """
    Process DNG files to JPG using RawTherapee.
    
    Args:
        batch_id: Batch identifier
        dng_files: List of DNG file records from database
        config: Processing configuration
        job_id: SLURM job ID
    
    Returns:
        List of processing events
    """
    logger.info(f"Processing {len(dng_files)} DNG files for batch {batch_id}")
    
    rt_cli = config['paths']['rawtherapee_cli']
    pp3_profile = config['paths']['pp3_profile']
    
    events = []
    
    output_dir = Path(config['paths']['output_base']) / batch_id / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    node_name = os.environ.get('SLURMD_NODENAME', 'unknown')
    
    for file_rec in dng_files:
        file_name = file_rec['file_name']
        input_path = Path(file_rec['root_path']) / file_rec['rel_path']
        output_path = output_dir / f"{Path(file_name).stem}.jpg"
        
        start_time = time.time()
        
        try:
            # Run RawTherapee CLI
            cmd = [
                rt_cli,
                '-o', str(output_path),
                '-p', pp3_profile,
                '-c', str(input_path),
                '-j100',  # JPEG quality
                '-Y',     # Overwrite
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per image
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                events.append({
                    'batch_id': batch_id,
                    'file_name': file_name,
                    'pipeline_stage': 'dng_to_jpg',
                    'status': 'success',
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'processing_time_sec': processing_time,
                    'error_message': None,
                    'job_id': job_id,
                    'node_name': node_name
                })
                
                logger.info(f"✓ {file_name} → {output_path.name} ({processing_time:.2f}s)")
            else:
                error_msg = result.stderr[:500]  # Truncate long errors
                
                events.append({
                    'batch_id': batch_id,
                    'file_name': file_name,
                    'pipeline_stage': 'dng_to_jpg',
                    'status': 'failed',
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'processing_time_sec': processing_time,
                    'error_message': error_msg,
                    'job_id': job_id,
                    'node_name': node_name
                })
                
                logger.error(f"✗ {file_name}: {error_msg}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            events.append({
                'batch_id': batch_id,
                'file_name': file_name,
                'pipeline_stage': 'dng_to_jpg',
                'status': 'failed',
                'input_path': str(input_path),
                'output_path': str(output_path) if output_path else None,
                'processing_time_sec': processing_time,
                'error_message': error_msg,
                'job_id': job_id,
                'node_name': node_name
            })
            
            logger.error(f"✗ {file_name}: {error_msg}")
    
    return events


# ============================================================
# Main Processing Logic
# ============================================================

def process_batch_pipeline(
    batch_id: str,
    pipeline_stage: str,
    config: Dict,
    job_id: str = None
) -> None:
    """
    Process a single batch through the specified pipeline stage.
    
    Args:
        batch_id: Batch identifier
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        config: Processing configuration
        job_id: SLURM job ID
    """
    logger.info(f"=" * 80)
    logger.info(f"Processing batch: {batch_id}")
    logger.info(f"Pipeline stage: {pipeline_stage}")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"=" * 80)
    
    with get_db_connection() as conn:
        # Sync inventory from globus_file_index
        logger.info("Syncing batch inventory from database...")
        sync_batch_inventory(conn, batch_id)
        
        # Mark as started
        start_batch_processing(conn, batch_id, pipeline_stage, job_id)
        conn.commit()
        
        try:
            if pipeline_stage == 'raw_to_dng':
                # Load required configuration
                camera_tags = load_camera_tags(Path(config['paths']['svs_tags']))
                color_matrix = np.load(config['paths']['color_matrix'], allow_pickle=True)
                
                # Get RAW files from database
                raw_files = get_batch_files(conn, batch_id, 'upload_raw', 'raw')
                
                if not raw_files:
                    logger.warning(f"No RAW files found for batch {batch_id}")
                    complete_batch_processing(conn, batch_id, pipeline_stage, success=False,
                                            error_message="No RAW files found")
                    conn.commit()
                    return
                
                # Process RAW → DNG
                events = process_raw_to_dng(
                    batch_id, raw_files, config, camera_tags, color_matrix, job_id
                )
                
            elif pipeline_stage == 'dng_to_jpg':
                # Get DNG files from database
                dng_files = get_batch_files(conn, batch_id, 'developed_jpg', 'dng')
                
                if not dng_files:
                    logger.warning(f"No DNG files found for batch {batch_id}")
                    complete_batch_processing(conn, batch_id, pipeline_stage, success=False,
                                            error_message="No DNG files found")
                    conn.commit()
                    return
                
                # Process DNG → JPG
                events = process_dng_to_jpg(batch_id, dng_files, config, job_id)
            
            else:
                raise ValueError(f"Unknown pipeline stage: {pipeline_stage}")
            
            # Log all events to database
            from svs_raw_api.db_interface import log_batch_events
            log_batch_events(conn, events)
            
            # Determine if processing succeeded
            failed_count = sum(1 for e in events if e['status'] == 'failed')
            success = failed_count == 0
            
            error_msg = None if success else f"{failed_count} files failed"
            
            # Mark as completed
            complete_batch_processing(conn, batch_id, pipeline_stage, success, error_msg)
            
            # Re-sync inventory to pick up new files
            sync_batch_inventory(conn, batch_id)
            
            conn.commit()
            
            logger.info(f"Batch {batch_id} processing complete:")
            logger.info(f"  Total: {len(events)}")
            logger.info(f"  Success: {len(events) - failed_count}")
            logger.info(f"  Failed: {failed_count}")
            
        except Exception as e:
            logger.exception(f"Batch processing failed: {e}")
            complete_batch_processing(conn, batch_id, pipeline_stage, success=False,
                                     error_message=str(e)[:500])
            conn.commit()
            raise


def process_multiple_batches(
    pipeline_stage: str,
    config: Dict,
    limit: int = 10,
    job_id: str = None
) -> None:
    """
    Process multiple batches from the database queue.
    
    Args:
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        config: Processing configuration
        limit: Maximum number of batches to process
        job_id: SLURM job ID
    """
    with get_db_connection() as conn:
        batches = get_batches_for_processing(conn, pipeline_stage, limit)
    
    if not batches:
        logger.info(f"No batches found for {pipeline_stage}")
        return
    
    logger.info(f"Found {len(batches)} batches for {pipeline_stage}")
    
    for i, batch in enumerate(batches, 1):
        batch_id = batch['batch_id']
        logger.info(f"\n[{i}/{len(batches)}] Processing batch: {batch_id}")
        
        try:
            process_batch_pipeline(batch_id, pipeline_stage, config, job_id)
        except Exception as e:
            logger.error(f"Failed to process batch {batch_id}: {e}")
            continue


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Database-integrated batch processor'
    )
    parser.add_argument(
        '--config', required=True, type=Path,
        help='Path to configuration YAML'
    )
    parser.add_argument(
        '--stage', required=True, choices=['raw_to_dng', 'dng_to_jpg'],
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--batch-id', type=str,
        help='Process specific batch (optional)'
    )
    parser.add_argument(
        '--limit', type=int, default=10,
        help='Maximum number of batches to process (default: 10)'
    )
    parser.add_argument(
        '--job-id', type=str,
        help='SLURM job ID for logging'
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get job ID from environment if not provided
    job_id = args.job_id or os.environ.get('SLURM_JOB_ID')
    
    if args.batch_id:
        # Process specific batch
        process_batch_pipeline(args.batch_id, args.stage, config, job_id)
    else:
        # Process multiple batches from queue
        process_multiple_batches(args.stage, config, args.limit, job_id)


if __name__ == '__main__':
    main()
