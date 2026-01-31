"""
CEMImage: Base Image Class

This module provides the base Image class with common image operations
for medical imaging. Specific correction methods are implemented in subclasses.
"""

import numpy as np
import pydicom
import scipy.ndimage


class Image:
    """
    Base class for medical image representation.
    
    Provides loading, basic processing (smoothing, normalization, inversion),
    and can be subclassed to add specific correction algorithms.
    """
    
    def __init__(self, dicom_path=None, pixel_array=None):
        """
        Initialize the Image class.
        
        Args:
            dicom_path (str, optional): Path to a DICOM file.
            pixel_array (np.ndarray, optional): 2D array of pixel values.
        """
        self.pixel_array = None
        self.metadata = None

        if dicom_path:
            self._load_dicom(dicom_path)
        elif pixel_array is not None:
            self.pixel_array = np.array(pixel_array, dtype=float)
        else:
            raise ValueError("Either dicom_path or pixel_array must be provided.")
        
    def _load_dicom(self, path):
        """Loads a DICOM file."""
        try:
            dcm = pydicom.dcmread(path)
            self.pixel_array = dcm.pixel_array.astype(float)
            self.metadata = dcm
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file at {path}: {e}")

    def smooth(self, sigma=1.0):
        """
        Applies Gaussian smoothing to the image.
        
        Args:
            sigma (float): Standard deviation for Gaussian kernel.
            
        Returns:
            Image: A new Image instance with smoothed pixels.
        """
        if self.pixel_array is None:
            raise ValueError("No pixel data to smooth.")
            
        smoothed_pixels = scipy.ndimage.gaussian_filter(self.pixel_array, sigma=sigma)
        return self.__class__(pixel_array=smoothed_pixels)

    def normalize(self):
        """
        Normalizes pixel values to the range [0, 1].
        Modifies the pixel_array in place.
        
        Returns:
            self: For method chaining.
        """
        if self.pixel_array is None:
            return self
            
        min_val = np.min(self.pixel_array)
        max_val = np.max(self.pixel_array)
        
        if max_val - min_val == 0:
            self.pixel_array = np.zeros_like(self.pixel_array)
        else:
            self.pixel_array = (self.pixel_array - min_val) / (max_val - min_val)
            
        return self

    def invert(self):
        """
        Inverts the pixel values.
        If normalized (0-1), it does 1 - pixels.
        Otherwise, it does max - pixels.
        Modifies the pixel_array in place.
        
        Returns:
            self: For method chaining.
        """
        if self.pixel_array is None:
            return self
            
        max_val = np.max(self.pixel_array)
        min_val = np.min(self.pixel_array)
        
        if min_val >= 0 and max_val <= 1.0:
            self.pixel_array = 1.0 - self.pixel_array
        else:
            self.pixel_array = max_val - self.pixel_array + min_val
             
        return self

    def copy(self):
        """
        Creates a copy of the image.
        
        Returns:
            Image: A new Image instance with copied pixel data.
        """
        return self.__class__(pixel_array=self.pixel_array.copy())
