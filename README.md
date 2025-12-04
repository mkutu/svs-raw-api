# SVS RAW Processing Pipeline

> Professional Snakemake pipeline for automated RAW → DNG → JPG image processing on USDA SCINet Ceres HPC

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Complete end-to-end pipeline for processing Sony ARW/RAW images from SVS-Vistek cameras in a three-tier storage architecture on USDA SCINet:

```
NCSU NFS → JUNO Archive → Ceres Scratch → Process (RAW→DNG→JPG) → Archive
```

**Key Features:**
- 🚀 **Parallel Processing**: 12x speedup with SLURM job arrays
- 📊 **Database Tracking**: Complete audit trail of all operations
- 🔄 **Globus Integration**: Automated data transfers between storage tiers
- 🎯 **Professional Workflow**: Snakemake orchestration with checkpointing
- 📝 **Comprehensive Logging**: Per-image and batch-level logs
- ⚡ **HPC Optimized**: Designed for SCINet Ceres infrastructure

## Quick Start

### 1. Clone and Install

```bash
# Clone repository
cd ~/repos
git clone <your-repo-url> svs-raw-api
cd svs-raw-api

# Run one-time setup
bash scripts/setup_environment.sh
```

### 2. Configure

```bash
# Edit configuration for your setup
nano config/scinet.yaml

# Test configuration
python -m svs_raw_api.test_config
```

### 3. Process Images

```bash
# Complete pipeline (sync → transfer → process)
./scripts/workflow.sh full-pipeline MD_2025-10-22

# Or step-by-step
./scripts/workflow.sh sync MD_2025-10-22      # NCSU → JUNO
./scripts/workflow.sh transfer MD_2025-10-22  # JUNO → Ceres
./scripts/workflow.sh process MD_2025-10-22   # RAW → DNG → JPG
```

## Architecture

### Three-Tier Storage System

| Tier | Location | Purpose | Retention |
|------|----------|---------|-----------|
| **Primary** | NCSU NFS | Field upload point | Permanent |
| **Archive** | JUNO LTS | Long-term storage | Permanent |
| **Scratch** | Ceres /90daydata | Processing workspace | 90 days |

### Processing Workflow

```
Input: Sony ARW (RAW) files
  ↓
Stage 1: RAW → DNG (Adobe format)
  - Custom color calibration
  - Metadata embedding
  - Parallel processing (12 jobs)
  ↓
Stage 2: DNG → JPG (Final output)
  - RawTherapee processing
  - Color correction
  - Quality optimization
  ↓
Output: High-quality JPG images
```

### Performance

- **Per 100-image batch:**
  - Data transfer: 5-15 minutes
  - RAW → DNG → JPG: 20-30 minutes
  - **Total: ~40-60 minutes**

- **Resource allocation:**
  - 12 parallel SLURM jobs
  - 4 cores × 16GB RAM per job
  - Total: 48 cores, 192GB RAM

## Repository Structure

```
svs-raw-api/
├── README.md                   # This file
├── pyproject.toml             # Python package configuration
├── LICENSE                    # License information
│
├── src/svs_raw_api/           # Main Python package
│   ├── __init__.py
│   ├── core.py               # RAW → DNG conversion
│   ├── cli.py                # Command-line interface
│   └── utils.py              # Utility functions
│
├── scripts/                   # Pipeline management scripts
│   ├── workflow.sh           # Main workflow orchestrator
│   ├── globus_manager.py     # Globus transfer management
│   ├── db_manager.py         # Database operations
│   ├── setup_environment.sh  # One-time setup script
│   └── validate_rawtherapee.sh  # RawTherapee validation
│
├── slurm/                     # SLURM job scripts
│   ├── run_snakemake.sh      # Main Snakemake SLURM script
│   └── array_process.sh      # Alternative array job script
│
├── config/                    # Configuration files
│   ├── scinet.yaml           # SCINet Ceres configuration
│   ├── snakemake_config.yaml # Snakemake workflow config
│   └── globus_endpoints.yaml # Globus endpoint definitions
│
├── Snakefile                  # Snakemake workflow definition
│
├── docs/                      # Documentation
│   ├── SETUP.md              # Detailed setup guide
│   ├── USAGE.md              # Usage examples
│   ├── ARCHITECTURE.md       # System architecture
│   └── TROUBLESHOOTING.md    # Common issues
│
├── tests/                     # Test suite
│   ├── test_config.py
│   ├── test_conversion.py
│   └── test_workflow.py
│
└── data/                      # Processing profiles
    └── profiles/
        ├── svs_tags.yaml
        ├── MD_calibration_matrix_optimized.npy
        └── MD_shr661_raw16.pp3
```

## Installation

### Prerequisites

- Python 3.8+
- Access to USDA SCINet Ceres
- Globus account
- SLURM account: `dash_agir`

### Detailed Setup

See [docs/SETUP.md](docs/SETUP.md) for complete installation instructions.

```bash
# 1. Load modules
module load miniconda
source activate /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep

# 2. Install package
cd ~/repos/svs-raw-api
pip install -e .

# 3. Install Snakemake
pip install snakemake --break-system-packages

# 4. Verify installation
python -m svs_raw_api.test_config
bash scripts/test_setup.sh
```

## Usage

### Basic Workflow

```bash
# Check for new batches needing sync
./scripts/workflow.sh check-missing

# Process a specific batch
./scripts/workflow.sh full-pipeline MD_2025-10-22

# Monitor processing
squeue -u $USER -A dash_agir
tail -f $PROJECT/logs/snakemake_*.out
```

### Advanced Usage

```bash
# Batch processing
for batch in MD_2025-10-{22..25}; do
    ./scripts/workflow.sh process $batch
    sleep 30  # Stagger submissions
done

# Custom Snakemake options
snakemake --config batch_id=MD_2025-10-22 --cores 4 --dry-run

# Check database status
python scripts/db_manager.py --db $DB_PATH summary
```

See [docs/USAGE.md](docs/USAGE.md) for more examples.

## Configuration

### SCINet Configuration (config/scinet.yaml)

```yaml
paths:
  project_base: /project/dash_agir/matthew.kutugata
  scratch_base: /90daydata/dash_agir/data/semifield-upload
  output_dir: ${project_base}/semifield-developed-images
  logs_dir: ${project_base}/logs

processing:
  height: 3024
  width: 4032
  threads_per_image: 4
  
slurm:
  partition: short
  account: dash_agir
  max_parallel_jobs: 12
```

### Customization

- **Resource allocation**: Edit `slurm.max_parallel_jobs` in config
- **Processing profiles**: Modify files in `data/profiles/`
- **Storage locations**: Update `paths` in config

## Monitoring

### Check Job Status

```bash
# Active jobs
squeue -u $USER -A dash_agir

# Recent completed jobs
sacct -X -u $USER -A dash_agir --starttime=now-1day

# Pipeline status
./scripts/workflow.sh status
```

### View Logs

```bash
# SLURM output
cat $PROJECT/logs/snakemake_svs_raw_process_*.out

# Per-image logs
ls $PROJECT/semifield-developed-images/<batch-id>/logs/

# Database query
python scripts/db_manager.py --db $DB_PATH query "SELECT * FROM batches WHERE batch_id='MD_2025-10-22'"
```

## Troubleshooting

### Common Issues

**Package not found:**
```bash
pip install -e . --no-deps
```

**RawTherapee not found:**
```bash
bash scripts/validate_rawtherapee.sh
source scripts/rawtherapee_path.sh
```

**Globus authentication:**
```bash
globus login
globus whoami
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_conversion.py -v

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Check imports
isort src/ scripts/ tests/

# Lint
flake8 src/ scripts/
```

## Documentation

- **[Setup Guide](docs/SETUP.md)** - Complete installation instructions
- **[Usage Guide](docs/USAGE.md)** - Detailed usage examples
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common problems and solutions
- **[API Reference](docs/API.md)** - Python API documentation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- **Matthew Kutugata** - *Initial work* - matthew.kutugata@usda.gov

## Acknowledgments

- USDA SCINet for providing HPC resources
- NCSU for primary data storage
- Globus for data transfer infrastructure

## Support

- **Issues**: Open an issue on GitHub
- **Email**: matthew.kutugata@usda.gov
- **SCINet**: scinet_vrsc@usda.gov

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready
