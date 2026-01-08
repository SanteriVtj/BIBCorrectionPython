"""
CEMImage: BIB Artifact Correction

This module provides the BIBCorrectedImage class which extends the base Image
class with breast-in-breast (BIB) artifact correction capabilities.
"""

import numpy as np
from .image import Image


class BIBCorrectedImage(Image):
    """
    Image class with BIB artifact correction capabilities.
    
    Extends the base Image class with methods for segmenting breast tissue,
    computing distance transforms, and applying adaptive multiplicative
    correction based on distance-from-boundary profiles.
    """

    def segment_breast(self, threshold=None, invert_mask=None):
        """
        Segments the breast tissue from background using thresholding.
        
        Args:
            threshold (float, optional): Manual threshold. If None, uses Otsu's method.
            invert_mask (bool, optional): If True, assumes breast is darker than background.
                                          If None, auto-detects based on which region is larger.
            
        Returns:
            np.ndarray: Binary mask where True indicates breast tissue.
        """
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
        
        if self.pixel_array is None:
            raise ValueError("No pixel data available.")
        
        # Normalize for thresholding
        img_normalized = (self.pixel_array - np.min(self.pixel_array)) / \
                         (np.max(self.pixel_array) - np.min(self.pixel_array))
        
        if threshold is None:
            threshold = threshold_otsu(img_normalized)
        
        # Create binary mask (initially assumes breast is brighter)
        mask = img_normalized > threshold
        
        # Auto-detect if we need to invert
        if invert_mask is None:
            invert_mask = np.sum(mask) > 0.5 * mask.size
        
        if invert_mask:
            mask = img_normalized < threshold
        
        # Morphological cleanup
        mask = binary_fill_holes(mask)
        structure = np.ones((5, 5))
        mask = binary_opening(mask, structure=structure)
        mask = binary_closing(mask, structure=structure)
        
        return mask

    def compute_distance_transform(self, mask=None):
        """
        Computes the Euclidean distance transform from each pixel to the boundary.
        
        Args:
            mask (np.ndarray, optional): Binary mask of the breast. If None, segments first.
            
        Returns:
            np.ndarray: Distance transform (same shape as pixel_array).
        """
        from scipy.ndimage import distance_transform_edt
        
        if mask is None:
            mask = self.segment_breast()
        
        return distance_transform_edt(mask)

    def extract_distance_profile(self, distances=None, mask=None, num_bins=100):
        """
        Extracts the median intensity as a function of distance from the boundary.
        
        Args:
            distances (np.ndarray, optional): Distance transform. Computed if None.
            mask (np.ndarray, optional): Binary mask. Computed if None.
            num_bins (int): Number of distance bins.
            
        Returns:
            tuple: (bin_centers, median_intensities, std_intensities)
        """
        if mask is None:
            mask = self.segment_breast()
        if distances is None:
            distances = self.compute_distance_transform(mask)
        
        # Get intensities and distances within the mask
        mask_flat = mask.flatten()
        distances_flat = distances.flatten()[mask_flat]
        intensities_flat = self.pixel_array.flatten()[mask_flat]
        
        # Bin by distance
        max_dist = np.max(distances_flat)
        bin_edges = np.linspace(0, max_dist, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        median_intensities = np.zeros(num_bins)
        std_intensities = np.zeros(num_bins)
        
        for i in range(num_bins):
            in_bin = (distances_flat >= bin_edges[i]) & (distances_flat < bin_edges[i+1])
            if np.sum(in_bin) > 0:
                median_intensities[i] = np.median(intensities_flat[in_bin])
                std_intensities[i] = np.std(intensities_flat[in_bin])
            else:
                median_intensities[i] = np.nan
                std_intensities[i] = np.nan
        
        # Fill NaNs with linear interpolation
        valid = ~np.isnan(median_intensities)
        if np.sum(valid) > 1:
            median_intensities = np.interp(bin_centers, bin_centers[valid], median_intensities[valid])
            std_intensities = np.interp(bin_centers, bin_centers[valid], std_intensities[valid])
        
        return bin_centers, median_intensities, std_intensities

    def compute_adaptive_correction_field(self, spline_order=3, smoothing=None, num_bins=100):
        """
        Computes the adaptive multiplicative correction field based on distance-from-boundary.
        
        Args:
            spline_order (int): Order of the smoothing spline (1-5). Lower = smoother.
            smoothing (float, optional): Smoothing factor for spline. None = auto.
            num_bins (int): Number of distance bins for profile extraction.
            
        Returns:
            tuple: (correction_field_2d, profile_data)
                   profile_data = (bin_centers, raw_correction, smooth_correction)
        """
        from scipy.interpolate import UnivariateSpline, interp1d
        
        # 1. Segment and compute distance transform
        mask = self.segment_breast()
        distances = self.compute_distance_transform(mask)
        
        # 2. Extract distance profile
        bin_centers, median_intensities, _ = self.extract_distance_profile(
            distances, mask, num_bins
        )
        
        # 3. Compute target intensity from interior region
        interior_start = int(num_bins * 0.2)
        interior_end = int(num_bins * 0.8)
        target_intensity = np.median(median_intensities[interior_start:interior_end])
        
        # 4. Compute raw correction factor: C(d) = target / observed(d)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_correction = target_intensity / median_intensities
            raw_correction[~np.isfinite(raw_correction)] = 1.0
        
        # Clip extreme values
        raw_correction = np.clip(raw_correction, 0.5, 2.0)
        
        # 5. Spline smoothing
        k = min(spline_order, len(bin_centers) - 1)
        k = max(k, 1)
        spline = UnivariateSpline(bin_centers, raw_correction, k=k, s=smoothing)
        smooth_correction = np.clip(spline(bin_centers), 0.5, 2.0)
        
        # 6. Map 1D correction back to 2D using distance transform
        correction_lookup = interp1d(
            bin_centers, smooth_correction,
            kind='linear',
            fill_value=(smooth_correction[0], smooth_correction[-1]),
            bounds_error=False
        )
        
        correction_field_2d = correction_lookup(distances)
        correction_field_2d[~mask] = 1.0
        
        profile_data = (bin_centers, raw_correction, smooth_correction)
        return correction_field_2d, profile_data

    def correct(self, spline_order=3, smoothing=None, num_bins=100):
        """
        High-level method to correct the breast-in-breast artifact.
        
        Args:
            spline_order (int): Order of the smoothing spline (1-5). Lower = smoother.
            smoothing (float, optional): Smoothing factor for spline.
            num_bins (int): Number of distance bins.
            
        Returns:
            tuple: (corrected_image, correction_field, profile_data)
        """
        field, profile_data = self.compute_adaptive_correction_field(
            spline_order=spline_order,
            smoothing=smoothing,
            num_bins=num_bins
        )
        
        corrected_pixels = self.pixel_array * field
        corrected_image = BIBCorrectedImage(pixel_array=corrected_pixels)
        
        return corrected_image, field, profile_data
