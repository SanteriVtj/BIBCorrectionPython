# CEMImage: Contrast-Enhanced Mammogram Artifact Correction

A Python library for processing contrast-enhanced mammograms (CEM), with specialized support for correcting the **breast-in-breast (BIB) artifact** using an adaptive, geometry-aware algorithm.

## Table of Contents
- [Background](#background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Mathematical Algorithm](#mathematical-algorithm)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Background

### Contrast-Enhanced Mammography
A contrast-enhanced mammogram is computed as a subtraction image:

$$I_{RC} = \ln I_{HE} - b \cdot \ln I_{LE}$$

Where:
- $I_{RC}$ = Recombined (subtraction) image
- $I_{HE}$ = High-energy X-ray exposure (45–49 kVp)
- $I_{LE}$ = Low-energy X-ray exposure (up to 33–35 kVp)
- $b$ = Subtraction factor (function of tissue thickness)

### The Breast-in-Breast Artifact
Due to non-uniform breast compression and differential scatter absorption, the recombined image exhibits:
- **Hyperintense band** near the tissue-background boundary
- **Hypointense plateau** near the center of the tissue

---

## Installation

```bash
conda create -n BIB python=3.10
conda activate BIB
conda install pydicom scipy numpy scikit-image matplotlib
```

---

## Quick Start

```python
from bib_correction import BIBCorrectedImage

# Load a DICOM image
img = BIBCorrectedImage(dicom_path="path/to/mammogram.dcm")

# Apply BIB artifact correction
corrected, field, profile = img.correct(spline_order=2)

# Normalize for display
corrected.normalize()
```

---

## Architecture

The library uses an **inheritance pattern** for extensibility:

```
Image (base class)
├── smooth(), normalize(), invert()
└── BIBCorrectedImage (subclass)
    ├── segment_breast()
    ├── compute_distance_transform()
    ├── extract_distance_profile()
    └── correct()
```

**Files:**
- `image.py` - Base `Image` class with common operations
- `bib_correction.py` - `BIBCorrectedImage` with BIB correction methods

**Adding new correction methods:** Create a new subclass of `Image`:
```python
from image import Image

class MyCustomCorrectedImage(Image):
    def correct(self, ...):
        # Your correction logic
        pass
```

---

## Mathematical Algorithm

### Overview
The correction models the BIB artifact as a function of **distance from the breast boundary**.

### Steps

1. **Segmentation**: Extract breast region $\Omega$ using Otsu thresholding
2. **Distance Transform**: $d(x,y) = \min_{(x',y') \in \partial\Omega} \|(x,y) - (x',y')\|$
3. **Profile Extraction**: $\tilde{I}(d_k) = \text{median}\{I(x,y) : d_k \leq d(x,y) < d_{k+1}\}$
4. **Target Intensity**: $I_{target} = \text{median}\{\tilde{I}(d_k) : 0.2d_{max} < d_k < 0.8d_{max}\}$
5. **Correction Factor**: $C_{raw}(d) = I_{target} / \tilde{I}(d)$
6. **Spline Smoothing**: $C_{smooth}(d) = \text{Spline}_k(C_{raw})$, clipped to $[0.5, 2.0]$
7. **2D Mapping**: $C_{2D}(x,y) = C_{smooth}(d(x,y))$
8. **Correction**: $I_{corrected} = I \cdot C_{2D}$

---

## API Reference

### Base Class: `Image` (image.py)

```python
Image(dicom_path=None, pixel_array=None)
```

| Method | Description |
|--------|-------------|
| `smooth(sigma)` | Gaussian smoothing, returns new Image |
| `normalize()` | Scales to [0, 1], in-place |
| `invert()` | Inverts pixel values, in-place |
| `copy()` | Creates a copy |

### Subclass: `BIBCorrectedImage` (bib_correction.py)

```python
from bib_correction import BIBCorrectedImage
```

| Method | Description |
|--------|-------------|
| `segment_breast(threshold, invert_mask)` | Binary mask of breast tissue |
| `compute_distance_transform(mask)` | Distance to boundary |
| `extract_distance_profile(distances, mask, num_bins)` | Intensity vs. distance |
| `correct(spline_order, smoothing, num_bins)` | Full correction pipeline |

**Parameters for `correct()`:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `spline_order` | 3 | Spline order (1-5). Lower = smoother. |
| `smoothing` | None | Smoothing factor. None = auto. |
| `num_bins` | 100 | Number of distance bins. |

**Returns:** `(corrected_image, correction_field_2d, profile_data)`

---

## Examples

### Full Workflow
```python
from bib_correction import BIBCorrectedImage
import matplotlib.pyplot as plt

img = BIBCorrectedImage(dicom_path="I248")
corrected, field, (bins, raw, smooth) = img.correct(spline_order=2, num_bins=50)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img.pixel_array, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(corrected.pixel_array, cmap='gray')
axes[1].set_title('Corrected')
plt.show()
```

---

## License
MIT License
