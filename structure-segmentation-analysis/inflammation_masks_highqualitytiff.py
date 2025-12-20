#!/usr/bin/env python3
"""
Combined Inflammation Mask Generator with Optional Parameter Tuning
Generates inflammation masks from immunofluorescence markers using maximum intensity projection,
Gaussian filtering, and thresholding for IgG4, AAV, and NK2 datasets.

Enhanced version with high-quality TIFF outputs (600+ DPI) and separate mask overlay file.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
from pathlib import Path
from skimage import io, filters
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import warnings
import seaborn as sns
from PIL import Image
warnings.filterwarnings('ignore')

class InflammationMaskGenerator:
    def __init__(self, base_image_dir, output_dir, csv_path=None, sigma=50, threshold_percentile=95, 
                 output_dpi=600, background_percentile=10):
        """
        Initialize the inflammation mask generator.
        
        Parameters:
        -----------
        base_image_dir : str
            Base directory containing raw image files
        output_dir : str
            Directory to save output masks and plots
        csv_path : str, optional
            Path to CSV file (only used for cell density calculations)
        sigma : float
            Sigma value for Gaussian filter (10-25 recommended, lower = less smoothing)
        threshold_percentile : float
            Percentile for automatic thresholding (0-100)
        output_dpi : int
            DPI for output TIFF files (default: 600)
        background_percentile : float
            Percentile for background removal before smoothing (default: 10)
            Removes weak non-specific signals to prevent false positives
        """
        self.base_image_dir = Path(base_image_dir)
        self.output_dir = Path(output_dir)
        self.csv_path = csv_path
        self.sigma = sigma
        self.threshold_percentile = threshold_percentile
        self.output_dpi = output_dpi
        self.background_percentile = background_percentile
        
        # Inflammation markers to include
        self.inflammation_markers = ['CD3', 'CD4', 'CD8', 'CD11c', 'CD16', 'CD20', 'CD68', 'SLAMF7']
        
        # Possible fluorescence channels (in order of preference)
        self.fluorescence_channels = ['AF488', 'Cy5', 'Atto550', 'AF750']
        
        # All datasets to process
        self.datasets = ['IgG4', 'AAV', 'NK2']
        
        # Possible cycle values (0 to 20)
        self.cycles = list(range(21))  # 0 to 20
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV data if provided (for density calculations)
        self.df = self._load_csv_data() if csv_path else None
        
        # Discover all available combinations from directory structure
        self.available_combinations = self._discover_combinations()
        
    def _load_csv_data(self):
        """Load CSV data if provided (for density calculations)."""
        if self.csv_path and Path(self.csv_path).exists():
            print(f"Loading CSV data from {self.csv_path}")
            return pd.read_csv(self.csv_path)
        return None
    
    def _discover_combinations(self):
        """
        Discover all available Dataset/Sample/Area combinations from directory structure.
        
        Returns:
        --------
        list : List of dictionaries with Dataset, Sample, Area combinations
        """
        print(f"Scanning directory structure: {self.base_image_dir}")
        combinations = []
        
        # Scan for each dataset
        for dataset in self.datasets:
            dataset_dir = self.base_image_dir / dataset
            if not dataset_dir.exists():
                print(f"Dataset directory not found: {dataset_dir}")
                continue
            
            # Look for Normalized_composites directory
            normalized_dir = dataset_dir / "Normalized_composites"
            if not normalized_dir.exists():
                print(f"Normalized_composites directory not found in {dataset_dir}")
                continue
            
            print(f"Scanning {normalized_dir}")
            
            # Check if directory has any subdirectories
            sample_dirs = [d for d in normalized_dir.iterdir() if d.is_dir()]
            print(f"  Found {len(sample_dirs)} sample directories")
            
            # Scan for samples
            for sample_dir in sample_dirs:
                sample = sample_dir.name
                print(f"  Checking sample: {sample}")
                
                # Scan for areas
                area_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
                print(f"    Found {len(area_dirs)} area directories")
                
                for area_dir in area_dirs:
                    area = area_dir.name
                    print(f"    Checking area: {area}")
                    
                    # Check if this combination has any inflammation markers
                    marker_count = 0
                    found_markers = []
                    
                    for marker in self.inflammation_markers:
                        marker_files = self._find_marker_files(area_dir, marker, dataset, sample, area)
                        if marker_files:
                            marker_count += 1
                            found_markers.append(marker)
                    
                    if marker_count >= 2:  # Need at least 2 markers
                        combinations.append({
                            'Dataset': dataset,
                            'Sample': sample,
                            'Area': area,
                            'MarkerCount': marker_count,
                            'Markers': found_markers
                        })
                        print(f"  Found: {dataset}/{sample}/{area} with {marker_count} markers: {found_markers}")
                    else:
                        print(f"    Skipping {dataset}/{sample}/{area}: only {marker_count} markers found")
        
        print(f"\nTotal combinations found: {len(combinations)}")
        
        # Print summary by dataset
        print("\nSummary by dataset:")
        for dataset in self.datasets:
            dataset_combos = [c for c in combinations if c['Dataset'] == dataset]
            print(f"  {dataset}: {len(dataset_combos)} combinations")
            
        return combinations
    
    def _find_marker_files(self, area_dir, marker, dataset, sample, area):
        """
        Find all files for a specific marker in an area directory.
        File pattern: {marker_name}_{fluorescence}_{cycle}_{dataset}_{CompName}.tif
        where CompName = {Sample}_{Area}
        """
        matching_files = []
        comp_name = f"{sample}_{area}"
        
        # Try both .tif and .tiff extensions
        extensions = ['.tif', '.tiff']
        
        for ext in extensions:
            # Try all combinations of fluorescence channels and cycles
            for fluorescence in self.fluorescence_channels:
                for cycle in self.cycles:
                    # Pattern: {marker}_{fluorescence}_{cycle}_{dataset}_{sample}_{area}.tif
                    pattern = f"{marker}_{fluorescence}_{cycle}_{dataset}_{comp_name}{ext}"
                    potential_files = list(area_dir.glob(pattern))
                    matching_files.extend(potential_files)
            
            # If no exact matches, try more flexible patterns
            if not matching_files:
                # Try pattern without specific cycle/fluorescence
                pattern = f"{marker}_*_{dataset}_{comp_name}{ext}"
                potential_files = list(area_dir.glob(pattern))
                matching_files.extend(potential_files)
            
            # If still no matches, try very flexible pattern
            if not matching_files:
                pattern = f"{marker}_*{ext}"
                potential_files = list(area_dir.glob(pattern))
                # Filter to only those that contain the dataset and comp_name
                matching_files = [f for f in potential_files 
                                if dataset in f.name and comp_name in f.name]
        
        return matching_files
    
    def _find_marker_image(self, dataset, sample, area, marker):
        """
        Find the image file for a specific marker.
        File pattern: {marker_name}_{fluorescence}_{cycle}_{dataset}_{CompName}.tif
        where CompName = {Sample}_{Area}
        
        Returns the path to the image file, or None if not found.
        """
        # Construct the area directory path
        dataset_dir = self.base_image_dir / dataset
        
        # Find Normalized_composites directory
        normalized_dir = dataset_dir / "Normalized_composites"
        if not normalized_dir.exists():
            return None
        
        area_dir = normalized_dir / str(sample) / str(area)
        if not area_dir.exists():
            return None
        
        comp_name = f"{sample}_{area}"
        extensions = ['.tif', '.tiff']
        
        # Try to find the marker file with different fluorescence channels and cycles
        # Prioritize by fluorescence channel preference
        for ext in extensions:
            for fluorescence in self.fluorescence_channels:
                for cycle in self.cycles:
                    # Pattern: {marker}_{fluorescence}_{cycle}_{dataset}_{sample}_{area}.tif
                    pattern = f"{marker}_{fluorescence}_{cycle}_{dataset}_{comp_name}{ext}"
                    matching_files = list(area_dir.glob(pattern))
                    
                    if matching_files:
                        return matching_files[0]  # Return first match with preferred fluorescence
        
        # If no exact matches found, try more flexible patterns
        for ext in extensions:
            pattern_variations = [
                f"{marker}_*_{dataset}_{comp_name}{ext}",    # Missing specific fluorescence/cycle
                f"{marker}_*{ext}"                           # Very flexible
            ]
            
            for pattern in pattern_variations:
                matching_files = list(area_dir.glob(pattern))
                if matching_files:
                    # Filter to ensure dataset and comp_name are in filename
                    filtered_files = [f for f in matching_files 
                                    if dataset in f.name and comp_name in f.name]
                    if filtered_files:
                        # Prefer certain fluorescence channels if multiple files
                        for channel in self.fluorescence_channels:
                            for file in filtered_files:
                                if f"_{channel}_" in file.name:
                                    return file
                        return filtered_files[0]  # Return first if no preferred channel
        
        # Last resort: case-insensitive search
        for ext in extensions:
            for file in area_dir.glob(f"*{ext}"):
                if (marker.lower() in file.name.lower() and 
                    dataset.lower() in file.name.lower() and 
                    comp_name.lower() in file.name.lower()):
                    return file
                
        return None
    
    def _load_marker_images(self, dataset, sample, area):
        """
        Load all inflammation marker images for a given dataset/sample/area combination.
        
        Returns:
        --------
        dict : Dictionary mapping marker names to image arrays
        """
        images = {}
        missing_markers = []
        found_files = []
        
        for marker in self.inflammation_markers:
            image_path = self._find_marker_image(dataset, sample, area, marker)
            
            if image_path and image_path.exists():
                try:
                    img = io.imread(str(image_path))
                    # Ensure image is 2D
                    if img.ndim > 2:
                        img = np.max(img, axis=0) if img.shape[0] < img.shape[-1] else img[:,:,0]
                    images[marker] = img
                    found_files.append(image_path.name)
                except Exception as e:
                    print(f"Error loading {marker} for {dataset}/{sample}/{area}: {e}")
                    missing_markers.append(marker)
            else:
                missing_markers.append(marker)
        
        print(f"Dataset {dataset}, Sample {sample}, Area {area}:")
        print(f"  Found markers: {list(images.keys())}")
        print(f"  Found files: {found_files}")
        if missing_markers:
            print(f"  Missing markers: {missing_markers}")
        
        return images
    
    def _create_max_intensity_projection(self, images):
        """
        Create maximum intensity projection from multiple marker images.
        
        Parameters:
        -----------
        images : dict
            Dictionary mapping marker names to image arrays
            
        Returns:
        --------
        np.array : Maximum intensity projection image
        """
        if not images:
            return None
        
        # Get image shape from first image
        first_img = list(images.values())[0]
        height, width = first_img.shape
        
        # Stack all images
        image_stack = []
        for marker, img in images.items():
            # Resize if necessary and normalize
            if img.shape != (height, width):
                from skimage.transform import resize
                img = resize(img, (height, width), preserve_range=True)
            
            # Normalize to 0-1 range
            img_norm = rescale_intensity(img.astype(np.float32), out_range=(0, 1))
            image_stack.append(img_norm)
        
        # Create maximum intensity projection
        image_stack = np.stack(image_stack, axis=0)
        max_projection = np.max(image_stack, axis=0)
        
        return max_projection
    
    def _create_inflammation_mask(self, max_projection, manual_threshold=None, global_threshold=None):
        """
        Create inflammation mask from maximum intensity projection.
        Uses pre-filtering to remove weak background before Gaussian smoothing
        to prevent false-positive amplification of non-specific signals.
        
        Parameters:
        -----------
        max_projection : np.array
            Maximum intensity projection image
        manual_threshold : float, optional
            Manual threshold value. If None, uses percentile-based thresholding.
        global_threshold : float, optional
            Pre-calculated global threshold value from all datasets combined.
            
        Returns:
        --------
        tuple : (filtered_image, mask, threshold_used, background_threshold)
        """
        # STEP 1: Pre-filter to remove weak background noise
        # This prevents Gaussian smoothing from amplifying non-specific weak signals
        nonzero_values = max_projection[max_projection > 0]
        if len(nonzero_values) > 0:
            background_threshold = np.percentile(nonzero_values, self.background_percentile)
            # Remove pixels below background threshold
            max_projection_cleaned = np.where(max_projection > background_threshold, max_projection, 0)
        else:
            background_threshold = 0
            max_projection_cleaned = max_projection.copy()
        
        # STEP 2: Apply Gaussian filter to cleaned image
        # Now smoothing won't amplify the removed weak signals
        filtered = filters.gaussian(max_projection_cleaned, sigma=self.sigma)
        
        # STEP 3: Determine threshold for final mask
        if manual_threshold is not None:
            threshold = manual_threshold
        elif global_threshold is not None:
            threshold = global_threshold
        else:
            # Local threshold (per-area) - old behavior
            threshold = np.percentile(filtered[filtered > 0], self.threshold_percentile)
        
        # STEP 4: Create binary mask
        mask = filtered > threshold
        
        return filtered, mask, threshold, background_threshold
    
    def _compute_cell_densities(self, dataset, sample, area, mask):
        """
        Compute cell densities within the inflammation mask.
        Only works if CSV data is available.
        
        Parameters:
        -----------
        dataset, sample, area : str
            Identifiers for the region
        mask : np.array
            Binary inflammation mask
            
        Returns:
        --------
        dict : Cell densities for different cell types, or empty dict if no CSV
        """
        if self.df is None:
            return {}
        
        # Filter data for this specific region
        region_data = self.df[
            (self.df['Dataset'] == dataset) & 
            (self.df['Sample'] == sample) & 
            (self.df['CompName'] == area)
        ]
        
        if region_data.empty:
            return {}
        
        # Get mask area (in pixels)
        mask_area_pixels = np.sum(mask)
        
        # Get cell type columns (excluding intensity and coordinate columns)
        cell_type_columns = [col for col in region_data.columns 
                           if not any(x in col for x in ['Int-mean', 'Centroid', 'Area', 'Diameter', 'eccentricity', 'Unnamed'])]
        
        # Calculate densities
        densities = {}
        for col in cell_type_columns:
            if col in region_data.columns and region_data[col].dtype in [np.float64, np.int64]:
                cell_count = region_data[col].sum()
                density = cell_count / mask_area_pixels if mask_area_pixels > 0 else 0
                densities[col] = density
        
        return densities
    
    def _save_tiff_with_dpi(self, image_array, filepath, dpi=None):
        """
        Save image as TIFF with specified DPI metadata.
        
        Parameters:
        -----------
        image_array : np.array
            Image data to save
        filepath : Path or str
            Output file path
        dpi : int, optional
            DPI value (default: self.output_dpi)
        """
        if dpi is None:
            dpi = self.output_dpi
        
        # Convert to PIL Image for proper DPI metadata
        if image_array.dtype != np.uint8:
            # Normalize to 0-255 range if not already uint8
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = rescale_intensity(image_array, out_range=(0, 255)).astype(np.uint8)
        
        # Handle different array dimensions
        if image_array.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(image_array, mode='L')
        elif image_array.ndim == 3:
            # RGB or RGBA
            if image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, mode='RGB')
            elif image_array.shape[2] == 4:
                pil_image = Image.fromarray(image_array, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {image_array.shape[2]}")
        else:
            raise ValueError(f"Unsupported array dimensions: {image_array.ndim}")
        
        # Save with DPI metadata
        pil_image.save(str(filepath), dpi=(dpi, dpi), compression='tiff_lzw')
    
    def _create_overlay_image(self, max_projection, mask):
        """
        Create RGB overlay image with mask highlighted.
        
        Parameters:
        -----------
        max_projection : np.array
            Maximum intensity projection (grayscale, 0-1 range)
        mask : np.array
            Binary mask
            
        Returns:
        --------
        np.array : RGB image with mask overlay (uint8, 0-255 range)
        """
        # Normalize max projection to 0-255 range
        proj_normalized = rescale_intensity(max_projection, out_range=(0, 255)).astype(np.uint8)
        
        # Create RGB image (all channels start as grayscale)
        overlay = np.stack([proj_normalized, proj_normalized, proj_normalized], axis=-1)
        
        # Highlight mask in green channel
        overlay[:, :, 1] = np.where(mask, 255, overlay[:, :, 1])
        
        return overlay
    
    def _save_results(self, dataset, sample, area, max_projection, filtered_image, mask, threshold, densities, background_threshold=None):
        """Save mask and results to files."""
        # Create subdirectory for this sample
        sample_output_dir = self.output_dir / f"{dataset}_{sample}_{area}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save mask as high-quality TIFF
        mask_path = sample_output_dir / "inflammation_mask.tiff"
        self._save_tiff_with_dpi(mask.astype(np.uint8) * 255, mask_path)
        
        # Save filtered image as high-quality TIFF
        filtered_path = sample_output_dir / "filtered_image.tiff"
        self._save_tiff_with_dpi(filtered_image, filtered_path)
        
        # Save max projection as high-quality TIFF
        projection_path = sample_output_dir / "max_projection.tiff"
        self._save_tiff_with_dpi(max_projection, projection_path)
        
        # Create and save mask overlay as separate high-quality TIFF
        overlay = self._create_overlay_image(max_projection, mask)
        overlay_path = sample_output_dir / "mask_overlay.tiff"
        self._save_tiff_with_dpi(overlay, overlay_path)
        
        # Create visualization PNG (includes all panels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(max_projection, cmap='gray')
        axes[0, 0].set_title('Max Intensity Projection')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(filtered_image, cmap='gray')
        title_str = f'Gaussian Filtered (σ={self.sigma})'
        if background_threshold is not None:
            title_str += f'\nBg removed: <{background_threshold:.3f}'
        axes[0, 1].set_title(title_str)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title(f'Inflammation Mask (threshold={threshold:.3f})')
        axes[1, 0].axis('off')
        
        # Use the same overlay in the visualization
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Mask Overlay')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'{dataset} - {sample} - {area}')
        plt.tight_layout()
        
        plot_path = sample_output_dir / "visualization.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save densities to CSV
        if densities:
            density_df = pd.DataFrame([densities])
            density_df['Dataset'] = dataset
            density_df['Sample'] = sample
            density_df['Area'] = area
            density_df['MaskAreaPixels'] = np.sum(mask)
            density_df['Threshold'] = threshold
            density_df['Sigma'] = self.sigma
            density_df['BackgroundThreshold'] = background_threshold if background_threshold is not None else 0
            density_df['BackgroundPercentile'] = self.background_percentile
            
            density_path = sample_output_dir / "cell_densities.csv"
            density_df.to_csv(str(density_path), index=False)
        
        print(f"Results saved to {sample_output_dir}")
        print(f"  - High-quality TIFFs saved at {self.output_dpi} DPI")
        
        return {
            'mask_path': mask_path,
            'filtered_path': filtered_path,
            'projection_path': projection_path,
            'overlay_path': overlay_path,
            'plot_path': plot_path,
            'densities': densities
        }
    
    def calculate_global_threshold(self, max_samples_per_dataset=None):
        """
        Calculate a global threshold using all areas from all datasets combined.
        This ensures that inflammatory datasets (IgG4, AAV) will have larger masks
        than normal datasets (NK2).
        
        Parameters:
        -----------
        max_samples_per_dataset : int, optional
            Maximum number of samples per dataset to use for threshold calculation
            (to speed up computation)
            
        Returns:
        --------
        float : Global threshold value
        """
        print("\n" + "="*60)
        print("CALCULATING GLOBAL THRESHOLD ACROSS ALL DATASETS")
        print("="*60)
        
        all_filtered_values = []
        combinations_to_use = self.available_combinations.copy()
        
        # Optionally limit samples per dataset for faster computation
        if max_samples_per_dataset:
            limited_combinations = []
            for dataset in self.datasets:
                dataset_combos = [c for c in combinations_to_use if c['Dataset'] == dataset]
                limited_combinations.extend(dataset_combos[:max_samples_per_dataset])
            combinations_to_use = limited_combinations
        
        print(f"Using {len(combinations_to_use)} areas to calculate global threshold:")
        
        # Collect filtered intensity values from all areas
        for combination in tqdm(combinations_to_use, desc="Loading and filtering images"):
            dataset = combination['Dataset']
            sample = combination['Sample']
            area = combination['Area']
            
            try:
                # Load marker images
                images = self._load_marker_images(dataset, sample, area)
                
                if len(images) < 2:
                    continue
                
                # Create maximum intensity projection
                max_projection = self._create_max_intensity_projection(images)
                
                if max_projection is None:
                    continue
                
                # Apply Gaussian filter with background removal
                nonzero_values = max_projection[max_projection > 0]
                if len(nonzero_values) > 0:
                    background_threshold = np.percentile(nonzero_values, self.background_percentile)
                    max_projection_cleaned = np.where(max_projection > background_threshold, max_projection, 0)
                else:
                    max_projection_cleaned = max_projection.copy()
                
                filtered = filters.gaussian(max_projection_cleaned, sigma=self.sigma)
                
                # Collect non-zero filtered values
                filtered_nonzero = filtered[filtered > 0]
                if len(filtered_nonzero) > 0:
                    # Sample values to avoid memory issues (take every Nth value)
                    sample_step = max(1, len(filtered_nonzero) // 10000)
                    all_filtered_values.extend(filtered_nonzero[::sample_step].tolist())
                
            except Exception as e:
                print(f"Error processing {dataset}/{sample}/{area}: {e}")
                continue
        
        if not all_filtered_values:
            print("ERROR: No valid filtered values collected!")
            return None
        
        # Calculate global threshold from all collected values
        all_filtered_values = np.array(all_filtered_values)
        global_threshold = np.percentile(all_filtered_values, self.threshold_percentile)
        
        print(f"\nGlobal threshold calculation complete:")
        print(f"  Total filtered values collected: {len(all_filtered_values):,}")
        print(f"  Threshold percentile: {self.threshold_percentile}")
        print(f"  Global threshold value: {global_threshold:.6f}")
        print(f"  Min filtered value: {all_filtered_values.min():.6f}")
        print(f"  Max filtered value: {all_filtered_values.max():.6f}")
        print(f"  Mean filtered value: {all_filtered_values.mean():.6f}")
        
        # Show distribution by dataset (if we can infer it)
        print("\nThis global threshold will be applied uniformly to all areas.")
        print("Expected behavior:")
        print("  - IgG4/AAV (inflammatory): High intensity → Large mask areas")
        print("  - NK2 (normal): Low intensity → Small mask areas")
        
        return global_threshold
    
    def process_all_samples(self, max_samples=None, target_datasets=None, use_global_threshold=True):
        """
        Process all discovered Dataset/Sample/Area combinations.
        
        Parameters:
        -----------
        max_samples : int, optional
            Maximum number of samples to process (for testing)
        target_datasets : list, optional
            List of specific datasets to process. If None, processes all available.
        use_global_threshold : bool
            Whether to use global threshold across all datasets
        """
        combinations_to_process = self.available_combinations
        
        # Filter by target datasets if specified
        if target_datasets:
            combinations_to_process = [c for c in combinations_to_process 
                                     if c['Dataset'] in target_datasets]
            print(f"Filtering to datasets: {target_datasets}")
        
        if max_samples:
            combinations_to_process = combinations_to_process[:max_samples]
        
        # Calculate global threshold if requested
        global_threshold = None
        if use_global_threshold:
            print("\nCalculating global threshold across all datasets...")
            global_threshold = self.calculate_global_threshold()
            if global_threshold is None:
                print("WARNING: Global threshold calculation failed. Falling back to local thresholds.")
                use_global_threshold = False
        
        results_summary = []
        
        print(f"\nProcessing {len(combinations_to_process)} combinations...")
        
        for combination in tqdm(combinations_to_process):
            dataset = combination['Dataset']
            sample = combination['Sample']
            area = combination['Area']
            
            try:
                # Load marker images from files
                images = self._load_marker_images(dataset, sample, area)
                
                if len(images) < 2:  # Need at least 2 markers
                    print(f"Skipping {dataset}/{sample}/{area}: insufficient markers ({len(images)} found)")
                    continue
                
                # Create maximum intensity projection
                max_projection = self._create_max_intensity_projection(images)
                available_markers = list(images.keys())
                
                if max_projection is None:
                    continue
                
                # Create inflammation mask
                filtered_image, mask, threshold, background_threshold = self._create_inflammation_mask(
                    max_projection, 
                    global_threshold=global_threshold if use_global_threshold else None
                )
                
                # Compute cell densities (if CSV data available)
                densities = self._compute_cell_densities(dataset, sample, area, mask)
                
                # Save results
                result = self._save_results(dataset, sample, area, max_projection, 
                                          filtered_image, mask, threshold, densities, background_threshold)
                
                # Add to summary
                summary_row = {
                    'Dataset': dataset,
                    'Sample': sample,
                    'Area': area,
                    'MarkersFound': len(available_markers),
                    'MarkersList': available_markers,
                    'MaskAreaPixels': np.sum(mask),
                    'MaskAreaPercent': (np.sum(mask) / mask.size) * 100,
                    'Threshold': threshold,
                    'BackgroundThreshold': background_threshold,
                    'Success': True
                }
                summary_row.update(result)
                results_summary.append(summary_row)
                
            except Exception as e:
                print(f"Error processing {dataset}/{sample}/{area}: {e}")
                import traceback
                traceback.print_exc()
                results_summary.append({
                    'Dataset': dataset,
                    'Sample': sample,
                    'Area': area,
                    'Error': str(e),
                    'Success': False
                })
        
        # Save overall summary
        summary_df = pd.DataFrame(results_summary)
        summary_path = self.output_dir / "processing_summary.csv"
        summary_df.to_csv(str(summary_path), index=False)
        print(f"Processing summary saved to {summary_path}")
        
        return summary_df


class ParameterTuner:
    def __init__(self, base_image_dir, output_dir, csv_path=None, output_dpi=600, background_percentile=10):
        self.base_image_dir = base_image_dir
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dpi = output_dpi
        self.background_percentile = background_percentile
        
        # Get a test sample by creating a temporary generator
        self.test_sample = self._get_test_sample()
        
    def _get_test_sample(self):
        """Get a representative test sample for parameter tuning."""
        # Create a temporary generator to discover combinations
        temp_generator = InflammationMaskGenerator(
            base_image_dir=self.base_image_dir,
            output_dir=self.output_dir / "temp",
            csv_path=self.csv_path,
            sigma=50,
            threshold_percentile=95,
            output_dpi=self.output_dpi
        )
        
        # Get available combinations
        combinations = temp_generator.available_combinations
        
        if combinations:
            # Choose combination with good marker coverage
            combinations = sorted(combinations, key=lambda x: x['MarkerCount'], reverse=True)
            return combinations[0]
        else:
            print("No suitable combinations found for parameter tuning!")
            return None
    
    def test_parameter_combinations(self):
        """Test different sigma and threshold combinations."""
        if self.test_sample is None:
            print("No test sample available for parameter tuning!")
            return pd.DataFrame()
            
        print(f"Testing parameters on: {self.test_sample}")
        
        # Parameter ranges for sensitivity testing
        # IMPORTANT: Lower sigma values recommended (10-25) to avoid amplifying weak signals
        sigma_values = [5, 10, 15, 20, 25, 30]  # Reduced range, lower values
        threshold_percentiles = [80, 85, 90, 92, 95, 97, 99]  # Lower values = more sensitive
        
        results = []
        
        for sigma in sigma_values:
            for threshold_pct in threshold_percentiles:
                try:
                    # Create generator with these parameters
                    generator = InflammationMaskGenerator(
                        base_image_dir=self.base_image_dir,
                        output_dir=self.output_dir / f"test_sigma{sigma}_thresh{threshold_pct}",
                        csv_path=self.csv_path,
                        sigma=sigma,
                        threshold_percentile=threshold_pct,
                        output_dpi=self.output_dpi
                    )
                    
                    # Load marker images for test sample
                    dataset = self.test_sample['Dataset']
                    sample = self.test_sample['Sample'] 
                    area = self.test_sample['Area']
                    
                    images = generator._load_marker_images(dataset, sample, area)
                    
                    if len(images) >= 2:
                        # Create maximum intensity projection
                        max_projection = generator._create_max_intensity_projection(images)
                        
                        if max_projection is not None:
                            # Create mask with background removal
                            filtered_image, mask, threshold_value, background_threshold = generator._create_inflammation_mask(max_projection)
                            
                            # Calculate metrics
                            mask_area = np.sum(mask)
                            total_area = mask.size
                            mask_percentage = (mask_area / total_area) * 100
                            mean_intensity = np.mean(max_projection[max_projection > 0])
                            max_intensity = np.max(max_projection)
                            
                            results.append({
                                'Sigma': sigma,
                                'ThresholdPercentile': threshold_pct,
                                'ThresholdValue': threshold_value,
                                'MaskAreaPixels': mask_area,
                                'MaskPercentage': mask_percentage,
                                'MarkersUsed': len(images),
                                'MeanIntensity': mean_intensity,
                                'MaxIntensity': max_intensity,
                                'Sensitivity': 'High' if mask_percentage > 15 else 'Medium' if mask_percentage > 5 else 'Low'
                            })
                            
                            print(f"σ={sigma}, thresh={threshold_pct}%: {mask_percentage:.1f}% area covered")
                    
                except Exception as e:
                    print(f"Error with σ={sigma}, thresh={threshold_pct}%: {e}")
        
        return pd.DataFrame(results)
    
    def create_sensitivity_recommendations(self, results_df):
        """Create recommendations for sensitive mask generation."""
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap of mask coverage
        pivot_coverage = results_df.pivot(index='Sigma', columns='ThresholdPercentile', values='MaskPercentage')
        sns.heatmap(pivot_coverage, annot=True, fmt='.1f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Mask Coverage (% of total area)')
        axes[0,0].set_xlabel('Threshold Percentile')
        axes[0,0].set_ylabel('Sigma')
        
        # 2. Heatmap of threshold values
        pivot_thresh = results_df.pivot(index='Sigma', columns='ThresholdPercentile', values='ThresholdValue')
        sns.heatmap(pivot_thresh, annot=True, fmt='.3f', cmap='plasma', ax=axes[0,1])
        axes[0,1].set_title('Actual Threshold Values')
        axes[0,1].set_xlabel('Threshold Percentile')
        axes[0,1].set_ylabel('Sigma')
        
        # 3. Sigma vs Coverage
        scatter1 = axes[1,0].scatter(results_df['Sigma'], results_df['MaskPercentage'], 
                         c=results_df['ThresholdPercentile'], cmap='coolwarm', s=60)
        axes[1,0].set_xlabel('Sigma')
        axes[1,0].set_ylabel('Mask Coverage (%)')
        axes[1,0].set_title('Sigma vs Coverage (color = threshold %ile)')
        cbar1 = plt.colorbar(scatter1, ax=axes[1,0])
        cbar1.set_label('Threshold Percentile')
        
        # 4. Threshold vs Coverage
        scatter2 = axes[1,1].scatter(results_df['ThresholdPercentile'], results_df['MaskPercentage'],
                         c=results_df['Sigma'], cmap='viridis', s=60)
        axes[1,1].set_xlabel('Threshold Percentile')
        axes[1,1].set_ylabel('Mask Coverage (%)')
        axes[1,1].set_title('Threshold vs Coverage (color = sigma)')
        cbar2 = plt.colorbar(scatter2, ax=axes[1,1])
        cbar2.set_label('Sigma')
        
        plt.tight_layout()
        plot_path = self.output_dir / "parameter_sensitivity_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results_df)
        
        return recommendations, plot_path
    
    def _generate_recommendations(self, results_df):
        """Generate parameter recommendations based on results."""
        
        recommendations = {
            'high_sensitivity': {},
            'medium_sensitivity': {},
            'low_noise': {}
        }
        
        # High sensitivity: maximize coverage while avoiding extreme values
        high_sens = results_df[results_df['MaskPercentage'] >= results_df['MaskPercentage'].quantile(0.8)]
        if not high_sens.empty:
            best_high = high_sens.loc[high_sens['MaskPercentage'].idxmax()]
            recommendations['high_sensitivity'] = {
                'sigma': best_high['Sigma'],
                'threshold_percentile': best_high['ThresholdPercentile'],
                'expected_coverage': best_high['MaskPercentage'],
                'description': 'Maximum sensitivity - detects weak inflammation signals'
            }
        
        # Medium sensitivity: balanced approach
        medium_range = results_df[
            (results_df['MaskPercentage'] >= 5) & 
            (results_df['MaskPercentage'] <= 20) &
            (results_df['Sigma'] <= 50)
        ]
        if not medium_range.empty:
            # Pick one with good balance
            best_medium = medium_range.loc[medium_range.index[len(medium_range)//2]]
            recommendations['medium_sensitivity'] = {
                'sigma': best_medium['Sigma'],
                'threshold_percentile': best_medium['ThresholdPercentile'],
                'expected_coverage': best_medium['MaskPercentage'],
                'description': 'Balanced sensitivity - good compromise'
            }
        
        # Low noise: conservative approach
        conservative = results_df[
            (results_df['Sigma'] >= 35) & 
            (results_df['ThresholdPercentile'] >= 95)
        ]
        if not conservative.empty:
            best_conservative = conservative.loc[conservative['MaskPercentage'].idxmin()]
            recommendations['low_noise'] = {
                'sigma': best_conservative['Sigma'],
                'threshold_percentile': best_conservative['ThresholdPercentile'],
                'expected_coverage': best_conservative['MaskPercentage'],
                'description': 'Conservative - only strong inflammation signals'
            }
        
        return recommendations


def run_parameter_analysis(base_image_dir, output_dir="./parameter_analysis", csv_path=None, output_dpi=600, background_percentile=10):
    """Run complete parameter analysis."""
    
    tuner = ParameterTuner(base_image_dir, output_dir, csv_path, output_dpi, background_percentile)
    
    print("Testing parameter combinations...")
    results_df = tuner.test_parameter_combinations()
    
    if results_df.empty:
        print("No successful parameter combinations found!")
        return None
    
    print("\nGenerating recommendations...")
    recommendations, plot_path = tuner.create_sensitivity_recommendations(results_df)
    
    # Save results
    results_path = Path(output_dir) / "parameter_test_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save recommendations
    rec_path = Path(output_dir) / "recommendations.txt"
    with open(rec_path, 'w') as f:
        f.write("INFLAMMATION MASK PARAMETER RECOMMENDATIONS\n")
        f.write("=" * 50 + "\n\n")
        
        for category, params in recommendations.items():
            if params:
                f.write(f"{category.upper().replace('_', ' ')}:\n")
                f.write(f"  Sigma: {params['sigma']}\n")
                f.write(f"  Threshold Percentile: {params['threshold_percentile']}\n")
                f.write(f"  Expected Coverage: {params['expected_coverage']:.1f}%\n")
                f.write(f"  Description: {params['description']}\n\n")
    
    # Print recommendations
    print(f"\nRESULTS SAVED TO: {output_dir}")
    print(f"Detailed results: {results_path}")
    print(f"Visualizations: {plot_path}")
    print(f"Recommendations: {rec_path}")
    
    print("\nRECOMMENDATIONS FOR SENSITIVE MASKS:")
    print("=" * 40)
    
    for category, params in recommendations.items():
        if params:
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  --sigma {params['sigma']} --threshold_percentile {params['threshold_percentile']}")
            print(f"  Expected coverage: {params['expected_coverage']:.1f}%")
            print(f"  {params['description']}")
    
    return results_df, recommendations


def interactive_choice():
    """Interactive choice for parameter analysis."""
    print("\n" + "="*60)
    print("INFLAMMATION MASK GENERATOR")
    print("="*60)
    print("This script can:")
    print("1. Run parameter analysis (recommended for first-time use)")
    print("2. Generate masks with default parameters")
    print("3. Generate masks with custom parameters")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    parser = argparse.ArgumentParser(description='Generate inflammation masks from raw immunofluorescence images')
    parser.add_argument('--images', required=True, help='Base directory containing raw image files')
    parser.add_argument('--output', required=True, help='Output directory for masks and results')
    parser.add_argument('--csv', help='Path to CSV file (optional, for cell density calculations)')
    parser.add_argument('--sigma', type=float, default=15, help='Gaussian filter sigma (default: 15, recommended: 10-25)')
    parser.add_argument('--threshold_percentile', type=float, default=95, 
                       help='Percentile for automatic thresholding (default: 95)')
    parser.add_argument('--background_percentile', type=float, default=10,
                       help='Percentile for background removal before smoothing (default: 10)')
    parser.add_argument('--output_dpi', type=int, default=600,
                       help='DPI for output TIFF files (default: 600)')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to process (for testing)')
    parser.add_argument('--tune_parameters', action='store_true', 
                       help='Run parameter tuning analysis instead of processing')
    parser.add_argument('--datasets', nargs='+', choices=['IgG4','AAV', 'NK2'], 
                       help='Specific datasets to process (default: all available)')
    parser.add_argument('--skip_param_analysis', action='store_true',
                       help='Skip parameter analysis and use provided/default parameters')
    parser.add_argument('--local_threshold', action='store_true',
                       help='Use per-area thresholds instead of global threshold (old behavior)')
    
    args = parser.parse_args()
    
    if args.tune_parameters:
        # Run parameter tuning only
        print("Running parameter tuning analysis...")
        results, recommendations = run_parameter_analysis(args.images, args.output + "_parameter_analysis", 
                                                         args.csv, args.output_dpi, args.background_percentile)
        
        if recommendations and 'high_sensitivity' in recommendations and recommendations['high_sensitivity']:
            params = recommendations['high_sensitivity']
            print(f"\nTo use recommended high sensitivity parameters:")
            print(f"python {__file__} --images {args.images} --output {args.output} --sigma {params['sigma']} --threshold_percentile {params['threshold_percentile']} --output_dpi {args.output_dpi} --skip_param_analysis")
    else:
        # Initialize generator
        generator = InflammationMaskGenerator(
            base_image_dir=args.images,
            output_dir=args.output,
            csv_path=args.csv,
            sigma=args.sigma,
            threshold_percentile=args.threshold_percentile,
            output_dpi=args.output_dpi,
            background_percentile=args.background_percentile
        )
        
        print(f"Found {len(generator.available_combinations)} Dataset/Sample/Area combinations")
        print(f"Output TIFF files will be saved at {args.output_dpi} DPI")
        
        # Determine threshold mode
        use_global_threshold = not args.local_threshold
        if use_global_threshold:
            print("Using GLOBAL THRESHOLD mode (recommended for cross-dataset comparison)")
            print("  - Same threshold applied to all areas")
            print("  - IgG4/AAV (inflammatory) will have larger masks")
            print("  - NK2 (normal) will have smaller masks")
        else:
            print("Using LOCAL THRESHOLD mode (per-area thresholds)")
            print("  - Each area gets its own threshold")
            print("  - May not be suitable for cross-dataset comparison")
        
        # Print summary by dataset
        datasets_found = {}
        for combo in generator.available_combinations:
            dataset = combo['Dataset']
            if dataset not in datasets_found:
                datasets_found[dataset] = []
            datasets_found[dataset].append(combo)
        
        print("\nSummary by dataset:")
        for dataset, combos in datasets_found.items():
            print(f"  {dataset}: {len(combos)} combinations")
            for combo in combos[:3]:  # Show first 3 per dataset
                print(f"    {combo['Sample']}/{combo['Area']} - {combo['MarkerCount']} markers")
            if len(combos) > 3:
                print(f"    ... and {len(combos) - 3} more")
        
        # Process samples
        results = generator.process_all_samples(
            max_samples=args.max_samples,
            target_datasets=args.datasets,
            use_global_threshold=use_global_threshold
        )
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {results['Success'].sum()} samples")
        print(f"Failed: {(~results['Success']).sum()} samples")
        
        # Print results by dataset
        if 'Dataset' in results.columns:
            print("\nResults by dataset:")
            for dataset in ['IgG4', 'AAV', 'NK2']:
                dataset_results = results[results['Dataset'] == dataset]
                if not dataset_results.empty:
                    success_count = dataset_results['Success'].sum()
                    total_count = len(dataset_results)
                    print(f"  {dataset}: {success_count}/{total_count} successful")


if __name__ == "__main__":
    # Check if running interactively or with command line arguments
    if len(sys.argv) == 1:  # No command line arguments - interactive mode
        # Hardcoded paths - modify these for your environment
        base_image_dir = "/project/mclark/SCAMPI_datasets"
        output_dir = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/inflammation_masks_output"
        csv_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMono_cellcatname23.csv"
        
        print("=== INFLAMMATION MASK GENERATION FOR ALL DATASETS ===")
        print(f"Scanning: {base_image_dir}")
        print("Directory structure: /project/mclark/SCAMPI_datasets/{Dataset}/Normalized_composites/{Sample}/{Area}/")
        print("File pattern: {marker_name}_{fluorescence}_{cycle}_{dataset}_{Sample}_{Area}.tif")
        print("Datasets: IgG4, AAV, NK2")
        print("Fluorescence: AF488, Cy5, Atto550, AF750")
        print("Cycles: 0-20")
        print("Markers: CD3, CD4, CD8, CD11c, CD16, CD20, CD68, SLAMF7")
        print("Output: High-quality TIFF files at 600 DPI")
        print("\nTHRESHOLD MODE:")
        print("  GLOBAL threshold (recommended): Same threshold for all datasets")
        print("    → IgG4/AAV (inflammatory) will have LARGE mask areas")
        print("    → NK2 (normal) will have SMALL mask areas")
        print("\nNOISE REDUCTION:")
        print(f"  Background removal: Removes weakest 10% of signals before smoothing")
        print(f"  Recommended sigma: 10-25 (lower = less smoothing, less false positives)")
        print(f"  Sigma 35+ can amplify weak non-specific signals!")
        
        # Interactive choice
        choice = interactive_choice()
        output_dpi = 600  # Default DPI for high quality
        
        if choice == 1:
            # Run parameter analysis
            print("\n=== PARAMETER ANALYSIS ===")
            param_result = run_parameter_analysis(
                base_image_dir, 
                "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/parameter_analysis", 
                csv_path,
                output_dpi
            )
            
            if param_result is not None:
                results, recommendations = param_result
                if recommendations and 'high_sensitivity' in recommendations and recommendations['high_sensitivity']:
                    params = recommendations['high_sensitivity']
                    
                    # Ask if user wants to proceed with mask generation
                    proceed = input(f"\nProceed with mask generation using recommended parameters? (y/n): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        choice = 2  # Proceed to mask generation
                        sigma = params['sigma']
                        threshold_pct = params['threshold_percentile']
                        print(f"Using recommended parameters: sigma={sigma}, threshold_percentile={threshold_pct}")
                    else:
                        print("Parameter analysis complete. Run the script again to generate masks.")
                        sys.exit(0)
                else:
                    print("No recommendations generated. Using default parameters.")
                    choice = 2
                    sigma = 15
                    threshold_pct = 85
            else:
                print("Parameter analysis failed. Using default parameters.")
                choice = 2
                sigma = 15
                threshold_pct = 85
        
        if choice == 2:
            # Use default parameters
            sigma = 15
            threshold_pct = 85
            print(f"Using default parameters: sigma={sigma}, threshold_percentile={threshold_pct}")
        
        elif choice == 3:
            # Custom parameters
            try:
                print("\nRECOMMENDATIONS:")
                print("  - Sigma: 10-25 (lower = less smoothing, avoid 35+)")
                print("  - Threshold percentile: 85-95")
                print("  - Background percentile: 5-15 (removes weakest signals)")
                sigma = float(input("Enter sigma value (10-25 recommended): "))
                threshold_pct = float(input("Enter threshold percentile (85-95 recommended): "))
                background_pct = float(input("Enter background percentile (5-15 recommended, default 10): ") or "10")
                dpi_input = input(f"Enter output DPI (default 600): ").strip()
                if dpi_input:
                    output_dpi = int(dpi_input)
            except ValueError:
                print("Invalid input. Using default parameters.")
                sigma = 15
                threshold_pct = 85
                background_pct = 10
                output_dpi = 600
        
        if choice in [2, 3]:
            # Set default background percentile if not set
            if 'background_pct' not in locals():
                background_pct = 10
            
            # Generate masks
            print(f"\n=== MASK GENERATION FOR ALL DATASETS ===")
            print(f"Output TIFFs will be saved at {output_dpi} DPI")
            print(f"Using GLOBAL THRESHOLD across all datasets")
            print(f"  - IgG4/AAV (inflammatory): Expected LARGE mask areas")
            print(f"  - NK2 (normal): Expected SMALL mask areas")
            print(f"Background removal: Removing weakest {background_pct}% before smoothing")
            print(f"Gaussian smoothing: σ={sigma}")
            
            generator = InflammationMaskGenerator(
                base_image_dir=base_image_dir,
                output_dir=output_dir,
                csv_path=csv_path,
                sigma=sigma,
                threshold_percentile=threshold_pct,
                output_dpi=output_dpi,
                background_percentile=background_pct
            )
            
            print(f"Found {len(generator.available_combinations)} combinations to process:")
            
            # Show breakdown by dataset
            datasets_found = {}
            for combo in generator.available_combinations:
                dataset = combo['Dataset']
                if dataset not in datasets_found:
                    datasets_found[dataset] = []
                datasets_found[dataset].append(combo)
            
            print("\nFound combinations by dataset:")
            for dataset, combos in datasets_found.items():
                print(f"  {dataset}: {len(combos)} combinations")
                for combo in combos[:2]:  # Show first 2 per dataset
                    print(f"    {combo['Sample']}/{combo['Area']} - {combo['MarkerCount']} markers: {combo['Markers']}")
                if len(combos) > 2:
                    print(f"    ... and {len(combos) - 2} more")
            
            # Ask for confirmation
            proceed = input(f"\nProcess all {len(generator.available_combinations)} combinations? (y/n): ").strip().lower()
            if proceed in ['y', 'yes']:
                # Process all combinations with global threshold
                results = generator.process_all_samples(use_global_threshold=True)
                
                print(f"\n=== PROCESSING SUMMARY ===")
                print(f"Total combinations processed: {len(results)}")
                print(f"Successful: {results['Success'].sum()}")
                print(f"Failed: {(~results['Success']).sum()}")
                
                # Summary by dataset
                if 'Dataset' in results.columns:
                    print("\nResults by dataset:")
                    for dataset in ['IgG4', 'AAV', 'NK2']:
                        dataset_results = results[results['Dataset'] == dataset]
                        if not dataset_results.empty:
                            success_count = dataset_results['Success'].sum()
                            total_count = len(dataset_results)
                            print(f"  {dataset}: {success_count}/{total_count} successful")
            else:
                # Process test samples only
                print("Processing first 3 combinations for testing...")
                results = generator.process_all_samples(max_samples=3, use_global_threshold=True)
                print("Test processing complete!")
                
                print(f"\nTo process all combinations with these parameters:")
                print(f"python {__file__} --images {base_image_dir} --output {output_dir} --sigma {sigma} --threshold_percentile {threshold_pct} --background_percentile {background_pct} --output_dpi {output_dpi} --skip_param_analysis")
        
    else:
        # Run command line interface
        main()
