"""
Pipeline orchestration for end-to-end RAW processing.

This module provides high-level interfaces for processing images
through the complete RAW → DNG → JPG pipeline.
"""

from pathlib import Path
from typing import Optional, Union, Literal
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from .converter import RawConverter
from .developer import DngDeveloper


logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    input_path: Path
    dng_path: Optional[Path] = None
    jpg_path: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.input_path.name}: {self.error or 'OK'}"


class Pipeline:
    """
    Complete RAW processing pipeline: RAW → DNG → JPG.
    
    This class orchestrates the full conversion workflow, handling:
    - RAW to DNG conversion with color calibration
    - DNG to JPG development with RawTherapee
    - Batch processing with progress tracking
    - Error handling and recovery
    - Optional intermediate file cleanup
    
    Parameters
    ----------
    converter : RawConverter
        RAW to DNG converter instance
    developer : DngDeveloper
        DNG to JPG developer instance
    keep_dng : bool, default=True
        Whether to keep intermediate DNG files after JPG creation
    
    Examples
    --------
    >>> from svs_raw_api import Pipeline, RawConverter, DngDeveloper
    >>> from svs_raw_api.calibration import CameraProfile
    >>> 
    >>> # Load camera profile
    >>> profile = CameraProfile.load("config/cameras/svs_shr661.yaml")
    >>> 
    >>> # Create pipeline
    >>> converter = RawConverter(profile.color_matrix, profile.camera_config)
    >>> developer = DngDeveloper("/usr/bin/rawtherapee-cli")
    >>> pipeline = Pipeline(converter, developer)
    >>> 
    >>> # Process single file
    >>> result = pipeline.process_file(
    ...     "image.raw",
    ...     output_dir="processed/",
    ...     stages=["convert", "develop"]
    ... )
    >>> 
    >>> # Process directory
    >>> results = pipeline.process_directory(
    ...     "raw_images/",
    ...     "processed/",
    ...     parallel=True,
    ...     max_workers=4
    ... )
    """
    
    def __init__(
        self,
        converter: RawConverter,
        developer: DngDeveloper,
        keep_dng: bool = True
    ):
        self.converter = converter
        self.developer = developer
        self.keep_dng = keep_dng
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        stages: list[Literal["convert", "develop"]] = ["convert", "develop"],
        jpg_quality: int = 95,
        profile: Optional[Path] = None
    ) -> ProcessingResult:
        """
        Process a single RAW file through the pipeline.
        
        Parameters
        ----------
        input_path : str or Path
            Path to input RAW file
        output_dir : Path
            Output directory for processed files
        stages : list of str, default=["convert", "develop"]
            Which stages to run: "convert" (RAW→DNG), "develop" (DNG→JPG)
        jpg_quality : int, default=95
            JPEG quality for final output
        profile : Path, optional
            RawTherapee profile to use for development
        
        Returns
        -------
        ProcessingResult
            Result object with paths and status
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = ProcessingResult(input_path=input_path)
        
        try:
            # Stage 1: RAW → DNG
            if "convert" in stages:
                dng_path = output_dir / f"{input_path.stem}.dng"
                logger.info(f"Converting {input_path.name} → {dng_path.name}")
                
                result.dng_path = self.converter.convert(
                    raw_path=input_path,
                    output_path=dng_path
                )
            else:
                # Assume DNG already exists
                result.dng_path = output_dir / f"{input_path.stem}.dng"
                if not result.dng_path.exists():
                    raise FileNotFoundError(
                        f"DNG file not found: {result.dng_path}. "
                        "Run with stages=['convert'] first."
                    )
            
            # Stage 2: DNG → JPG
            if "develop" in stages:
                jpg_path = output_dir / f"{input_path.stem}.jpg"
                logger.info(f"Developing {result.dng_path.name} → {jpg_path.name}")
                
                result.jpg_path = self.developer.develop(
                    dng_path=result.dng_path,
                    output_path=jpg_path,
                    quality=jpg_quality,
                    profile=profile
                )
                
                # Cleanup intermediate DNG if requested
                if not self.keep_dng and result.dng_path.exists():
                    logger.debug(f"Removing intermediate DNG: {result.dng_path}")
                    result.dng_path.unlink()
                    result.dng_path = None
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {e}")
            result.error = str(e)
        
        return result
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        stages: list[Literal["convert", "develop"]] = ["convert", "develop"],
        pattern: str = "*.raw",
        jpg_quality: int = 95,
        profile: Optional[Path] = None,
        parallel: bool = False,
        max_workers: int = 4,
        progress: bool = True
    ) -> list[ProcessingResult]:
        """
        Process all RAW files in a directory.
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing RAW files
        output_dir : str or Path
            Output directory for processed files
        stages : list of str
            Which stages to run
        pattern : str, default="*.raw"
            Glob pattern for finding input files
        jpg_quality : int, default=95
            JPEG quality
        profile : Path, optional
            RawTherapee profile
        parallel : bool, default=False
            Whether to process files in parallel
        max_workers : int, default=4
            Number of parallel workers
        progress : bool, default=True
            Whether to show progress
        
        Returns
        -------
        list of ProcessingResult
            Results for each processed file
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find all input files
        input_files = sorted(input_dir.glob(pattern))
        
        if not input_files:
            logger.warning(f"No files matching '{pattern}' found in {input_dir}")
            return []
        
        logger.info(f"Processing {len(input_files)} files from {input_dir}")
        
        if not parallel:
            # Sequential processing
            results = []
            for i, input_file in enumerate(input_files, 1):
                if progress:
                    logger.info(f"[{i}/{len(input_files)}] {input_file.name}")
                
                result = self.process_file(
                    input_path=input_file,
                    output_dir=output_dir,
                    stages=stages,
                    jpg_quality=jpg_quality,
                    profile=profile
                )
                results.append(result)
            
            return results
        
        else:
            # Parallel processing
            results = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self.process_file,
                        input_file,
                        output_dir,
                        stages,
                        jpg_quality,
                        profile
                    ): input_file
                    for input_file in input_files
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    results.append(result)
                    
                    if progress:
                        status = "✓" if result.success else "✗"
                        logger.info(
                            f"[{completed}/{len(input_files)}] {status} "
                            f"{result.input_path.name}"
                        )
            
            return results
    
    def get_summary(self, results: list[ProcessingResult]) -> dict:
        """
        Generate summary statistics from processing results.
        
        Parameters
        ----------
        results : list of ProcessingResult
            Results from process_directory()
        
        Returns
        -------
        dict
            Summary statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        dngs_created = sum(1 for r in results if r.dng_path and r.dng_path.exists())
        jpgs_created = sum(1 for r in results if r.jpg_path and r.jpg_path.exists())
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'dngs_created': dngs_created,
            'jpgs_created': jpgs_created,
            'errors': [
                {'file': r.input_path.name, 'error': r.error}
                for r in results if not r.success
            ]
        }
    
    def print_summary(self, results: list[ProcessingResult]) -> None:
        """Print human-readable summary of processing results."""
        summary = self.get_summary(results)
        
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"Total files:     {summary['total']}")
        print(f"Successful:      {summary['successful']}")
        print(f"Failed:          {summary['failed']}")
        print(f"Success rate:    {summary['success_rate']:.1%}")
        print(f"DNGs created:    {summary['dngs_created']}")
        print(f"JPGs created:    {summary['jpgs_created']}")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for err in summary['errors'][:10]:  # Show first 10
                print(f"  ✗ {err['file']}: {err['error']}")
            
            if len(summary['errors']) > 10:
                print(f"  ... and {len(summary['errors']) - 10} more")
        
        print("="*60 + "\n")