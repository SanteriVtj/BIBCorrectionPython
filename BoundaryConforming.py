import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

def create_adaptive_mesh_correction(image):
    # Ensure image is in a format OpenCV likes (uint8 for thresholding/contours)
    # We normalize it for the mask generation process
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 1. Segment the breast
    _, mask = cv2.threshold(img_norm, 10, 255, cv2.THRESH_BINARY)
    
    # FIX: Corrected function name (camelCase) and return values
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No breast contour detected. Check image intensity.")
        
    # Take the largest contour
    breast_contour = max(contours, key=cv2.contourArea)

    # 2. Generate Mesh Nodes (The "Red Dots")
    # Sample points along the boundary (every 30th point for density)
    boundary_points = breast_contour[::30, 0, :] 
    
    # Sample center points to handle the "hypointense plateau"
    M = cv2.moments(breast_contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    interior_points = np.array([[cx, cy], [cx-40, cy], [cx+40, cy]])
    
    nodes = np.vstack([boundary_points, interior_points])
    
    # 3. Define Targets
    # We want to smooth the intensities toward the global tissue mean
    tissue_pixels = image[mask > 0]
    avg_intensity = np.mean(tissue_pixels)
    
    # Get values at nodes and calculate multiplicative correction targets
    node_values = np.array([image[int(p[1]), int(p[0])] for p in nodes])
    node_targets = avg_intensity / (node_values + 1e-6)

    # 4. Fit Radial Basis Function (Smooth Spline Field)
    # Using thin_plate_spline ensures C2 continuity (very smooth)
    rbf = RBFInterpolator(nodes, node_targets, kernel='thin_plate_spline')

    # 5. Apply to Full Image
    h, w = image.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    grid_points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
    
    flat_field = rbf(grid_points)
    correction_field = flat_field.reshape(h, w)
    
    # Multiplicative correction
    corrected_image = image * correction_field
    corrected_image[mask == 0] = 0 
    
    return corrected_image, correction_field, nodes

# --- Test with Synthetic Data ---
test_img = np.zeros((300, 300))
cv2.ellipse(test_img, (300, 150), (250, 120), 0, 0, 360, 120, -1)
# Add hyperintense edge artefact
cv2.ellipse(test_img, (300, 150), (245, 115), 0, 0, 360, 180, 5)

corrected, field, nodes = create_adaptive_mesh_correction(test_img)

# Visual Comparison
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(test_img, cmap='gray'); plt.title("Original (Artefact)")
plt.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10)
plt.subplot(132); plt.imshow(field, cmap='jet'); plt.title("Correction Field")
plt.subplot(133); plt.imshow(corrected, cmap='gray'); plt.title("Corrected Image")
plt.tight_layout()
plt.show()