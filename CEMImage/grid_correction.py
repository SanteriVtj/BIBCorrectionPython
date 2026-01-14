"""
CEMImage: Grid-Based Artifact Correction

This module provides the GridCorrectedImage class which extends the base Image
class with grid interpolation based correction.

The method:
1. Interpolate the masked image
"""

import numpy as np
try:
    from .image import Image
except ImportError:
    from image import Image

class GridCorrectedImage(Image):

    def detect_boundary(self, threshold=None):
        """
        Detects the breast boundary points.
        
        Args:
            threshold (float, optional): Manual threshold. If None, uses Otsu.
            
        Returns:
            tuple: (mask, boundary_points) where boundary_points is Nx2 array of (row, col)
        """
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
        from skimage.measure import find_contours
        
        if self.pixel_array is None:
            raise ValueError("No pixel data available.")
        
        # Normalize for thresholding
        img_normalized = self.normalize().pixel_array
        
        if threshold is None:
            threshold = threshold_otsu(img_normalized)
        
        # Create binary mask
        mask = img_normalized > threshold
        
        # Auto-detect inversion
        if np.sum(mask) > 0.5 * mask.size:
            mask = img_normalized < threshold
        
        # Morphological cleanup
        mask = binary_fill_holes(mask)
        structure = np.ones((5, 5))
        mask = binary_opening(mask, structure=structure)
        mask = binary_closing(mask, structure=structure)
        
        # Find contours (boundary)
        contours = find_contours(mask.astype(float), 0.5)
        
        if not contours:
            return mask, np.array([])
        
        # Take the longest contour (main breast boundary)
        boundary = max(contours, key=len)
        
        return mask, boundary
    
    def correct(self, grid_height=15, grid_width=15, kx=3, ky=3, smoothing_sigma=10, kernel_size=(15,15), clip_min=.5, clip_max=2):
        """
        Applies grid interpolation based correction for the BIB artefact.

        Args:
            grid_height (int, optional): number of knots in y-axis
            grid_width (int, optional): number of knots in x-axis
            kx (int, optional): degree of spline order for x-axis
            ky (int, optional): degree of spline order for y-axis
            smoothing_sigma (float, optional): gaussian smoothing sigma
            kernel_size: size of gaussian kernel
        """

        from scipy.interpolate import RectBivariateSpline
        import scipy.ndimage

        rows, cols = self.pixel_array.shape
        X = self.pixel_array

        mask, _ = self.detect_boundary()

        # Generate interpolation knots
        x_grid = np.linspace(0, cols-1, grid_width, dtype=int)
        y_grid = np.linspace(0, rows-1, grid_height, dtype=int)

        # Blur the original image
        X = scipy.ndimage.gaussian_filter(X, sigma=smoothing_sigma)*mask

        # Interpolate the blurred image
        itp = RectBivariateSpline(
            y_grid, 
            x_grid, 
            np.array([[X[r,c] for c in x_grid] for r in y_grid]), 
            kx=kx, 
            ky=ky
        )

        Z = np.clip(itp(range(rows), range(cols)), clip_min, clip_max)*mask

        with np.errstate(divide='ignore', invalid='ignore'):
            field = np.median(Z[mask])/Z
            field[~mask] = 0

        return GridCorrectedImage(pixel_array=field*self.pixel_array), field, itp