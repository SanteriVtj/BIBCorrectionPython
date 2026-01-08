"""
Verification script for adaptive BIB artifact correction on I248.
"""
import numpy as np
from bib_correction import BIBCorrectedImage
import matplotlib.pyplot as plt

# Load the example image
print("Loading I248...")
img = BIBCorrectedImage(dicom_path="I248")
print(f"Image shape: {img.pixel_array.shape}")
print(f"Original intensity range: {np.min(img.pixel_array):.1f} - {np.max(img.pixel_array):.1f}")

# Segment the breast
print("\nSegmenting breast...")
mask = img.segment_breast()
print(f"Mask coverage: {np.sum(mask) / mask.size * 100:.1f}%")

# Compute distance transform
print("\nComputing distance transform...")
distances = img.compute_distance_transform(mask)
print(f"Max distance from boundary: {np.max(distances):.1f} pixels")

# Extract distance profile
print("\nExtracting distance profile...")
bin_centers, median_intensities, std_intensities = img.extract_distance_profile(distances, mask)
print(f"Near-boundary median intensity: {median_intensities[0]:.1f}")
print(f"Deep-interior median intensity: {median_intensities[-1]:.1f}")

# Apply correction
print("\nApplying BIB artifact correction...")
corrected, field, profile_data = img.correct(spline_order=2, num_bins=50)

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

# Mask
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('Breast Mask')
axes[0, 1].axis('off')

# Distance transform
axes[0, 2].imshow(distances, cmap='viridis')
axes[0, 2].set_title('Distance Transform')
axes[0, 2].axis('off')

# Correction field
axes[1, 0].imshow(field, cmap='coolwarm', vmin=0.8, vmax=1.2)
axes[1, 0].set_title('Correction Field')
axes[1, 0].axis('off')

# Corrected
axes[1, 1].imshow(corrected.pixel_array, cmap='gray')
axes[1, 1].set_title('Corrected')
axes[1, 1].axis('off')

# Profile
bin_centers, raw_correction, smooth_correction = profile_data
axes[1, 2].plot(bin_centers, raw_correction, 'b-', alpha=0.5, label='Raw')
axes[1, 2].plot(bin_centers, smooth_correction, 'r-', linewidth=2, label='Smoothed')
axes[1, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
axes[1, 2].set_xlabel('Distance from boundary (pixels)')
axes[1, 2].set_ylabel('Correction factor')
axes[1, 2].set_title('Correction Profile')
axes[1, 2].legend()
axes[1, 2].set_ylim(0.5, 1.5)

plt.tight_layout()
plt.savefig('bib_correction_result.png', dpi=150)
print("\nSaved visualization to bib_correction_result.png")

print("\nVerification complete!")
