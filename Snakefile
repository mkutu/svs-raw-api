"""
SVS RAW Image Processing Pipeline
Snakemake workflow for USDA SciNet HPC (Ceres)

Converts Sony ARW/RAW images → DNG → JPG with parallel SLURM execution

Usage:
    # Process single batch
    snakemake --profile config/slurm --config batch_id=MD_2025-10-22

    # Or use the submission script
    sbatch slurm/submit_snakemake.sh MD_2025-10-22

Author: Matthew Kutugata
"""

import os
import sys
from pathlib import Path
import logging

# ============================================================================
# Configuration
# ============================================================================

configfile: "config/config.yaml"

# Get batch ID from config or command line
BATCH_ID = config.get("batch_id", "")
if not BATCH_ID:
    raise ValueError("batch_id must be specified via --config batch_id=<batch_id>")

# Paths from config
PATHS = config["paths"]
PROCESSING = config["processing"]
SLURM = config["slurm"]

# Input/Output directories
INPUT_DIR = Path(PATHS["scratch_dir"]) / BATCH_ID
OUTPUT_DIR = Path(PATHS["output_dir"]) / BATCH_ID
DNG_DIR = OUTPUT_DIR / "dngs"
JPG_DIR = OUTPUT_DIR / "images"
LOG_DIR = OUTPUT_DIR / "logs"

# ============================================================================
# Find RAW files
# ============================================================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Batch directory not found: {INPUT_DIR}")

RAW_FILES = list(INPUT_DIR.glob("*.RAW"))
if not RAW_FILES:
    RAW_FILES = list(INPUT_DIR.glob("*.ARW"))

if not RAW_FILES:
    raise FileNotFoundError(f"No RAW files found in {INPUT_DIR}")

# Extract basenames without extension
SAMPLES = [f.stem for f in RAW_FILES]

# Report what we found
print(f"\n{'='*70}")
print(f"SVS RAW Processing Pipeline - Batch: {BATCH_ID}")
print(f"{'='*70}")
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Files:  {len(SAMPLES)} RAW images")
print(f"{'='*70}\n")

# ============================================================================
# Rules
# ============================================================================

# Default target: all JPG files
rule all:
    input:
        jpgs = expand(str(JPG_DIR / "{sample}.jpg"), sample=SAMPLES),
        summary = str(OUTPUT_DIR / "processing_summary.txt")
    message: "Pipeline complete for batch {BATCH_ID}"


rule raw_to_dng:
    """Convert RAW to DNG format"""
    input:
        raw = lambda wildcards: next(f for f in RAW_FILES if f.stem == wildcards.sample)
    output:
        dng = DNG_DIR / "{sample}.dng"
    params:
        svs_tags = PATHS["svs_tags"],
        color_matrix = PATHS["color_matrix"],
        height = PROCESSING["height"],
        width = PROCESSING["width"],
        repo_root = PATHS["repo_root"]
    log:
        LOG_DIR / "raw_to_dng_{sample}.log"
    resources:
        mem_mb = SLURM["mem_per_job_mb"],
        runtime = SLURM["time_per_job_min"],
        tmpdir = "/tmp"
    threads: SLURM["cpus_per_job"]
    conda:
        "config/environment.yaml"
    script:
        "workflow/scripts/raw_to_dng.py"


rule dng_to_jpg:
    """Convert DNG to JPG using RawTherapee"""
    input:
        dng = DNG_DIR / "{sample}.dng"
    output:
        jpg = JPG_DIR / "{sample}.jpg"
    params:
        pp3_profile = PATHS["pp3_profile"],
        rawtherapee_cli = PATHS["rawtherapee_cli"],
        threads = PROCESSING["threads_per_image"]
    log:
        LOG_DIR / "dng_to_jpg_{sample}.log"
    resources:
        mem_mb = SLURM["mem_per_job_mb"],
        runtime = SLURM["time_per_job_min"]
    threads: PROCESSING["threads_per_image"]
    conda:
        "config/environment.yaml"
    script:
        "workflow/scripts/dng_to_jpg.py"


rule create_summary:
    """Generate processing summary report"""
    input:
        jpgs = expand(str(JPG_DIR / "{sample}.jpg"), sample=SAMPLES)
    output:
        summary = OUTPUT_DIR / "processing_summary.txt"
    params:
        batch_id = BATCH_ID,
        input_dir = INPUT_DIR,
        output_dir = OUTPUT_DIR,
        total_files = len(SAMPLES)
    run:
        import datetime
        from pathlib import Path

        # Count successful outputs
        dng_count = len(list(DNG_DIR.glob("*.dng")))
        jpg_count = len(list(JPG_DIR.glob("*.jpg")))

        # Calculate sizes
        def get_dir_size(path):
            total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            return total / (1024**3)  # GB

        input_size = get_dir_size(params.input_dir) if Path(params.input_dir).exists() else 0
        dng_size = get_dir_size(DNG_DIR) if DNG_DIR.exists() else 0
        jpg_size = get_dir_size(JPG_DIR) if JPG_DIR.exists() else 0

        # Write summary
        with open(output.summary, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SVS RAW Image Processing Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Batch ID:      {params.batch_id}\n")
            f.write(f"Completed:     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Dir:     {params.input_dir}\n")
            f.write(f"Output Dir:    {params.output_dir}\n\n")
            f.write("-"*70 + "\n")
            f.write("Processing Results\n")
            f.write("-"*70 + "\n")
            f.write(f"Total RAW files:      {params.total_files}\n")
            f.write(f"DNGs created:         {dng_count} / {params.total_files}\n")
            f.write(f"JPGs created:         {jpg_count} / {params.total_files}\n\n")
            f.write("-"*70 + "\n")
            f.write("Storage Summary\n")
            f.write("-"*70 + "\n")
            f.write(f"Input (RAW):          {input_size:.2f} GB\n")
            f.write(f"Output (DNG):         {dng_size:.2f} GB\n")
            f.write(f"Output (JPG):         {jpg_size:.2f} GB\n")
            f.write(f"Total output:         {dng_size + jpg_size:.2f} GB\n\n")
            f.write("-"*70 + "\n")
            f.write("Status\n")
            f.write("-"*70 + "\n")
            if jpg_count == params.total_files:
                f.write("SUCCESS: All files processed successfully\n")
            else:
                failed = params.total_files - jpg_count
                f.write(f"PARTIAL: {failed} files failed to process\n")
                f.write("Check individual log files for details\n")
            f.write("="*70 + "\n")

        # Print to console
        with open(output.summary, 'r') as f:
            print(f.read())


# Optional cleanup rule (not included in 'all' by default)
rule cleanup:
    """Clean up intermediate DNG files if JPGs were created successfully"""
    input:
        summary = OUTPUT_DIR / "processing_summary.txt"
    output:
        touch(OUTPUT_DIR / ".cleanup_done")
    params:
        keep_dngs = PROCESSING.get("keep_dngs", True)
    run:
        if not params.keep_dngs:
            import shutil
            print(f"Cleaning up DNG files from {DNG_DIR}")
            if DNG_DIR.exists():
                shutil.rmtree(DNG_DIR)
                print(f"✓ Removed {DNG_DIR}")
        else:
            print("Keeping DNG files (keep_dngs=True)")


# Success marker
onsuccess:
    print(f"\n{'='*70}")
    print(f"Pipeline completed successfully!")
    print(f"Batch: {BATCH_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


onerror:
    print(f"\n{'='*70}")
    print(f"Pipeline failed!")
    print(f"Batch: {BATCH_ID}")
    print(f"Check logs in: {LOG_DIR}")
    print(f"{'='*70}\n")
