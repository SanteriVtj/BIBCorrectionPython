
import numpy as np
import pytest
from medical_image import Image
import math

def test_initialization_pixel_array():
    data = np.zeros((10, 10))
    img = Image(pixel_array=data)
    assert img.pixel_array.shape == (10, 10)
    assert np.all(img.pixel_array == 0)

def test_smoothing():
    data = np.zeros((10, 10))
    data[5, 5] = 100
    img = Image(pixel_array=data)
    smoothed_img = img.smooth(sigma=1.0)
    
    # Center pixel should decrease
    assert smoothed_img.pixel_array[5, 5] < 100
    # Neighbor pixel should increase
    assert smoothed_img.pixel_array[5, 6] > 0
    # Original should be unchanged
    assert img.pixel_array[5, 5] == 100

def test_path_finding_flat_surface():
    # 3x3 flat surface
    data = np.zeros((3, 3))
    img = Image(pixel_array=data)
    
    start = (0, 0)
    end = (2, 2)
    path = img.compute_path(start, end, weight_type='3d_surface')
    
    # Path should exist
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Minimal length: 3 points for diagonal (0,0)->(1,1)->(2,2)
    assert len(path) == 3 
    assert (1, 1) in path

def test_path_finding_3d_valley():
    # ... (same as before) ...
    data = np.zeros((5, 5))
    data[2, 2] = 1000 
    
    img = Image(pixel_array=data)
    
    start = (2, 0)
    end = (2, 4)
    
    path = img.compute_path(start, end, weight_type='3d_surface')
    
    path_pixels = [data[r, c] for r, c in path]
    assert max(path_pixels) < 500

def test_generate_paths():
    """Test geometric path generation."""
    data = np.zeros((10, 10))
    img = Image(pixel_array=data)
    
    start_points = [(2, 0), (7, 0)]
    end_x = 9
    
    paths = img.generate_paths(start_points, end_x, num_steps=20)
    
    assert len(paths) == 2
    assert len(paths[0]) == 20
    assert paths[0][0, 1] == 0  # Start x
    assert paths[0][-1, 1] == 9 # End x
    assert abs(paths[0][0, 0] - 2) < 0.01 # Start y
    # End y should be distributed. 
    # For 2 points, targets might be evenly spaced. 
    
def test_correction_inhomogeneity():
    """Test correction on a synthetic gradient image."""
    # Create an image with a horizontal gradient (simulate inhomogeneity)
    # Brightness drops effectively from left to right
    rows, cols = 50, 50
    data = np.zeros((rows, cols))
    for c in range(cols):
        data[:, c] = 1000 * (1.0 - 0.5 * c / cols) # 1000 down to 500
        
    img = Image(pixel_array=data)
    
    # We expect correction to flatten this gradient.
    # The 'target' is median profile. 
    # If we correct, the std dev of rows should decrease?
    # Or at least, the difference between left and right should decrease.
    
    corrected, field = img.correct_inhomogeneity(num_paths=5)
    
    # Check original gradient
    orig_left = np.mean(img.pixel_array[:, 0])
    orig_right = np.mean(img.pixel_array[:, -1])
    orig_diff = abs(orig_left - orig_right)
    
    # Check corrected gradient
    # Note: 'correct_inhomogeneity' multiplies by field.
    # The field should be roughly inverse of profile.
    # If original is decreasing, field should be increasing.
    
    corr_left = np.mean(corrected.pixel_array[:, 0])
    corr_right = np.mean(corrected.pixel_array[:, -1])
    corr_diff = abs(corr_left - corr_right)
    
    print(f"Original Diff: {orig_diff}, Corrected Diff: {corr_diff}")
    
    assert corr_diff < orig_diff * 0.5 # Should reduce difference significantly

def test_normalization():
    data = np.array([[10, 20], [30, 40]])
    img = Image(pixel_array=data)
    img.normalize()
    
    assert np.isclose(np.min(img.pixel_array), 0.0)
    assert np.isclose(np.max(img.pixel_array), 1.0)
    assert np.isclose(img.pixel_array[0, 0], 0.0)
    assert np.isclose(img.pixel_array[1, 1], 1.0)
    assert np.isclose(img.pixel_array[0, 1], 1/3)

def test_inversion():
    data = np.array([[0.0, 0.25], [0.5, 1.0]])
    img = Image(pixel_array=data)
    img.invert()
    
    assert np.isclose(img.pixel_array[0, 0], 1.0)
    assert np.isclose(img.pixel_array[-1, -1], 0.0)
    assert np.isclose(img.pixel_array[0, 1], 0.75)

if __name__ == "__main__":
    try:
        test_initialization_pixel_array()
        print("test_initialization_pixel_array passed")
        test_smoothing()
        print("test_smoothing passed")
        test_normalization()
        print("test_normalization passed")
        test_inversion()
        print("test_inversion passed")
        test_path_finding_flat_surface()
        print("test_path_finding_flat_surface passed")
        test_path_finding_3d_valley()
        print("test_path_finding_3d_valley passed")
        test_generate_paths()
        print("test_generate_paths passed")
        test_correction_inhomogeneity()
        print("test_correction_inhomogeneity passed")
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
