"""
Verification script for path-based artifact correction on I248.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from path_correction import PathCorrectedImage
import matplotlib.pyplot as plt

# Load the example image
print("Loading I248...")
img = PathCorrectedImage(dicom_path="../Processed/I248")
print(f"Image shape: {img.pixel_array.shape}")

# Detect boundary
print("\nDetecting boundary...")
mask, boundary = img.detect_boundary()
print(f"Mask coverage: {np.sum(mask) / mask.size * 100:.1f}%")
print(f"Boundary points: {len(boundary)}")

# Generate paths
print("\nGenerating Bezier paths...")
paths, _ = img.generate_bezier_paths(num_paths=12)
print(f"Generated {len(paths)} paths")
if paths:
    print(f"Average path length: {np.mean([len(p) for p in paths]):.0f} points")

# Sampling is now handled inside correction or visualized later

# Apply correction with weighted edge sampling
print("\nApplying path-based correction...")
corrected, field, paths, sample_points = img.correct(
    num_paths=12, 
    dense_dist=150,    # Focus on first 150px (typical artifact width)
    dense_step=15,     # High fidelity at edge
    sparse_step=200,   # Sparse interior
    interpolation='cubic',
    smoothing_sigma=15
)

# Compute statistics
original_std = np.std(img.pixel_array[mask])
corrected_std = np.std(corrected.pixel_array[mask])
print(f"\nOriginal std within mask: {original_std:.1f}")
print(f"Corrected std within mask: {corrected_std:.1f}")
print(f"Std reduction: {(1 - corrected_std/original_std) * 100:.1f}%")

# Save visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(img.pixel_array, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Paths visualization
img.visualize_paths(paths, sample_points, ax=axes[0, 1])

# Mask
axes[0, 2].imshow(mask, cmap='gray')
axes[0, 2].set_title('Breast Mask')
axes[0, 2].axis('off')

# Correction field
im = axes[1, 0].imshow(field, cmap='coolwarm', vmin=0.8, vmax=1.2)
axes[1, 0].set_title('Correction Field')
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# Corrected
axes[1, 1].imshow(corrected.pixel_array, cmap='gray')
axes[1, 1].set_title('Corrected')
axes[1, 1].axis('off')

# Difference
diff = corrected.pixel_array - img.pixel_array
axes[1, 2].imshow(diff, cmap='RdBu', vmin=-np.percentile(np.abs(diff), 99), 
                  vmax=np.percentile(np.abs(diff), 99))
axes[1, 2].set_title('Difference (Corrected - Original)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('path_correction_result.png', dpi=150)
print("\nSaved visualization to path_correction_result.png")

print("\nVerification complete!")
