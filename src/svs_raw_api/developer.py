"""
DNG to JPG developer using RawTherapee.
Simple wrapper around RawTherapee CLI.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict

class DngToJpg:
    """Develop DNG files to JPG using RawTherapee."""
    
    def __init__(self, cfg: Dict):
        """
        Args:
            rt_cli: Path to rawtherapee-cli executable
            pp3_profile: Optional RawTherapee processing profile
            validate_script: Optional path to RawTherapee validation/install script
        """
        self.cfg = cfg
        self.rt_cli = Path(cfg["paths"]["rawtherapee_cli"])
        self.pp3_profile = Path(cfg["paths"]["pp3_profile"]) if cfg["paths"].get("pp3_profile") else None
        self.validate_script = Path(cfg["paths"]["rawtherapee_validate_script"]) if cfg["paths"].get("rawtherapee_validate_script") else None

        if not self.validate_installation():
            self.install_rawtherapee()
        
        if not self.rt_cli.exists():
            raise FileNotFoundError(f"RawTherapee CLI not found: {self.rt_cli}")
    
    def develop(self, dng_path: Path, jpg_path: Path, quality: int = 100) -> Path:
        """
        Develop DNG to JPG.
        
        Args:
            dng_path: Input DNG file
            jpg_path: Output JPG file
            quality: JPEG quality 0-100
        """
        dng_path = Path(dng_path)
        jpg_path = Path(jpg_path)
        
        if not dng_path.exists():
            raise FileNotFoundError(f"DNG not found: {dng_path}")
        
        try:
            # Build command
            cmd = [
                str(self.rt_cli),
                "-O", str(jpg_path),
                "-p", str(self.pp3_profile),
                f'-j{quality}', '-js3', '-Y', # Overwrite
                "-c", str(dng_path),
            ]

            threads_per_instance = self.cfg.get("processing", {}).get("threads_per_image", 1)
            max_threads = 50  # Total number of threads to use
            num_instances = 12  # Expected number of parallel threads
            threads_per_instance = max(1, max_threads // num_instances) #
            env = {
                **os.environ,
                "LANG": "en_US.UTF-8",
                "OMP_NUM_THREADS": str(threads_per_instance),
                "OMP_DYNAMIC": "TRUE",  # Allows OpenMP to optimize thread count
                "OMP_NESTED": "FALSE"  # Disables nested parallelism
            }
            
            # Run
            jpg_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"RawTherapee failed: {e.stderr}")
        
        return jpg_path
    
    def validate_installation(self) -> bool:
        """Check if RawTherapee CLI is accessible."""
        try:
            result = subprocess.run([str(self.rt_cli), '--version'], capture_output=True, text=True)
            return result.returncode in [0, 2]
        except FileNotFoundError:
            return False
        
    def install_rawtherapee(self):
        """Use the script in ./scripts/validate_rawtherapee.sh to install and unpack RawTherapee if needed."""
        if not self.validate_script or not self.validate_script.exists():
            raise FileNotFoundError("Validation script not found.")
        
        result = subprocess.run([str(self.validate_script)], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"RawTherapee installation failed: {result.stderr}")