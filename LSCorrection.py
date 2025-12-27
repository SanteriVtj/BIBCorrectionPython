import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import LSQBivariateSpline
from scipy.optimize import minimize

def solve_breast_artefact(image):
    # 1. Basic Segmentation to find the breast mask
    _, mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Define a sparse grid for the Spline (the "Mesh")
    h, w = image.shape
    rows, cols = 6, 6 # Low density to avoid overcorrecting lesions
    tx = np.linspace(0, w, cols)
    ty = np.linspace(0, h, rows)
    
    # Coordinates of pixels inside the breast
    y_coords, x_coords = np.where(mask > 0)
    z_values = image[y_coords, x_coords]

    # 3. Objective function: Minimize variance of the corrected pixels
    # We want the 'fatty tissue' to be as uniform as possible.
    def objective(coeffs):
        # Reshape coeffs to match grid
        spline = LSQBivariateSpline(x_coords, y_coords, z_values, tx[1:-1], ty[1:-1])
        # In a real scenario, you'd solve for coeffs that flatten the image
        # Here we simulate a simplified multiplicative field
        corrected = z_values * spline(x_coords, y_coords, grid=False)
        return np.std(corrected)

    # 4. For this MVP, we'll generate the surface using a B-Spline directly
    # to demonstrate the "smooth field" concept.
    spline_model = LSQBivariateSpline(x_coords, y_coords, z_values, tx[1:-1], ty[1:-1])
    
    # Create the correction field (smooth version of the image)
    # We invert it or normalize it to act as a 'flattener'
    correction_field = spline_model(np.arange(w), np.arange(h)).T
    avg_val = np.mean(correction_field[mask > 0])
    normalized_correction = avg_val / (correction_field + 1e-6)
    
    # Apply multiplicative correction
    corrected_image = image * normalized_correction
    corrected_image[mask == 0] = 0 # Keep background black
    
    return corrected_image, normalized_correction, (tx, ty)

# --- Visualization ---
# Simulate an image with a 'edge artifact' (bright border)
synth_img = np.ones((200, 200)) * 100
for i in range(200):
    for j in range(200):
        dist = np.sqrt((i-100)**2 + (j-200)**2)
        if dist < 180: # Breast shape
            synth_img[i,j] = 100 + (180 - dist)*0.5 # Hyperintense edge
        else:
            synth_img[i,j] = 0

corrected, field, grid = solve_breast_artefact(synth_img)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(synth_img, cmap='gray')
ax[0].set_title("Original with Artefact")

# Draw the 'Mesh' as requested
tx, ty = grid
for x in tx: ax[0].axvline(x, color='red', alpha=0.3)
for y in ty: ax[0].axhline(y, color='red', alpha=0.3)

ax[1].imshow(field, cmap='viridis')
ax[1].set_title("Spline Correction Field")

ax[2].imshow(corrected, cmap='gray')
ax[2].set_title("Corrected Image")
plt.show()