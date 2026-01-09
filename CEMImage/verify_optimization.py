"""
Verification script for Optimization-Based correction.
Compares Heuristic vs. Optimization methods.
"""
import matplotlib.pyplot as plt
import numpy as np
from path_correction import PathCorrectedImage

def verify_optimization():
    print("Loading I248...")
    img = PathCorrectedImage(dicom_path="../Processed/I248")
    
    # 1. Heuristic Correction (Baseline)
    print("\n[1] Running Heuristic Correction...")
    heuristic_kws = dict(
        num_paths=15, 
        dense_step=10, 
        sparse_step=150, 
        gradient_window=30,
        smoothing_sigma=15
    )
    
    corr_h, field_h, _, _ = img.correct(**heuristic_kws)
    
    mask, _ = img.detect_boundary()
    std_orig = np.std(img.pixel_array[mask])
    std_h = np.std(corr_h.pixel_array[mask])
    red_h = (std_orig - std_h) / std_orig * 100
    
    print(f"Heuristic Std: {std_h:.1f} (Reduction: {red_h:.1f}%)")
    
    # 2. Optimization Correction
    print("\n[2] Running Optimization Correction...")
    # Using same sampling parameters
    corr_opt, field_opt, _, _ = img.correct_optimized(
        **heuristic_kws,
        regularization_weight=0.1, # Lowered to favor std reduction over stability
        downsample_factor=4
    )
    
    std_opt = np.std(corr_opt.pixel_array[mask])
    red_opt = (std_orig - std_opt) / std_orig * 100
    
    print(f"Optimized Std: {std_opt:.1f} (Reduction: {red_opt:.1f}%)")
    
    # 3. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Images
    vmin, vmax = np.percentile(img.pixel_array, (1, 99))
    
    axes[0, 0].imshow(img.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Original (Std: {std_orig:.0f})")
    
    axes[0, 1].imshow(corr_h.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Heuristic (Red: {red_h:.1f}%)")
    
    axes[0, 2].imshow(corr_opt.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"Optimized (Red: {red_opt:.1f}%)")
    
    # Row 2: Fields
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(field_h, cmap='jet')
    axes[1, 1].set_title("Heuristic Field")
    
    axes[1, 2].imshow(field_opt, cmap='jet')
    axes[1, 2].set_title("Optimized Field")
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('optimization_result.png')
    print("\nSaved comparison to optimization_result.png")

if __name__ == "__main__":
    verify_optimization()
