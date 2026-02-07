"""
CEMImage: Path-Based Artifact Correction

This module provides the PathCorrectedImage class which extends the base Image
class with path-based correction using Bezier curves and adaptive sampling.

The method:
1. Detects breast boundary
2. Generates Bezier curves from boundary points toward the opposite edge
3. Samples points adaptively along each path (more points for longer paths)
4. Computes correction factors at sample points
5. Interpolates to create 2D correction field
"""

import numpy as np
try:
    from .image import Image
except ImportError:
    from image import Image


class PathCorrectedImage(Image):
    """
    Image class with path-based artifact correction using Bezier curves.
    
    This approach creates an interpolation grid from paths that follow
    the breast geometry, allowing for more flexible correction than
    distance-based methods.
    """
    def __init__(self, dicom_path=None, pixel_array=None, sigma=25, radius=50):
        super().__init__(dicom_path=dicom_path, pixel_array=pixel_array)
        self._blurred_pixel_array = None
        self._sigma = sigma
        self._radius = radius
        
    @property
    def blurred_pixel_array(self):
        if self._blurred_pixel_array is None:
            self._blurred_pixel_array = self._normalized_gaussian(sigma=self._sigma, radius=self._radius)
        return self._blurred_pixel_array
        
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
        
        # Normalize for thresholding (use copy to allow non-destructive)
        img_normalized = self.copy().normalize().pixel_array
        
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
    
    def compute_boundary_normals(self, boundary_points, smoothing=5, mask=None):
        """
        Computes inward-pointing normal vectors at boundary points.
        
        Args:
            boundary_points: Nx2 array of (row, col) coordinates
            smoothing: Window size for tangent estimation
            mask: Binary mask of the breast (used to orient normals)
            
        Returns:
            Nx2 array of normal vectors
        """
        n = len(boundary_points)
        normals = np.zeros_like(boundary_points, dtype=float)
        
        # Compute centroid if mask is provided
        if mask is not None:
            import scipy.ndimage
            centroid = np.array(scipy.ndimage.center_of_mass(mask))
        else:
            centroid = np.mean(boundary_points, axis=0)
            
        for i in range(n):
            # Use neighbors for tangent estimation
            i_prev = (i - smoothing) % n
            i_next = (i + smoothing) % n
            
            p_prev = boundary_points[i_prev]
            p_next = boundary_points[i_next]
            
            # Tangent vector
            tangent = p_next - p_prev
            
            # Normal: rotate 90 degrees (perpendicular)
            # Try [-dy, dx]
            normal = np.array([-tangent[1], tangent[0]])
            
            # Normalize
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            
            # Check orientation: should point towards centroid
            p_curr = boundary_points[i]
            to_centroid = centroid - p_curr
            
            if np.dot(normal, to_centroid) < 0:
                normal = -normal
            
            normals[i] = normal
        
        return normals

    def generate_bezier_paths(self, num_paths=15):
        """
        Generates non-overlapping Bezier paths that start normal to the skin
        and curb towards the image boundary (chest wall).
        """
        mask, boundary = self.detect_boundary()
        
        if len(boundary) == 0:
            return [], mask
            
        import scipy.ndimage
        
        rows, cols = self.pixel_array.shape
        
        # 1. Determine "Chest Wall" Side
        centroid = scipy.ndimage.center_of_mass(mask)
        margin = 5
        edges = {
            'left': np.sum(boundary[:, 1] < margin),
            'right': np.sum(boundary[:, 1] > cols - margin),
            'top': np.sum(boundary[:, 0] < margin),
            'bottom': np.sum(boundary[:, 0] > rows - margin)
        }
        chest_side = max(edges, key=edges.get)
        
        if chest_side == 'left':
            chest_wall_x = 0
            skin_points = boundary[boundary[:, 1] > margin * 2]
            # Normal flows towards left (negative column direction)
            # But normals computed are "inward". If breast is on Right, inward is Left.
        elif chest_side == 'right':
            chest_wall_x = cols - 1
            skin_points = boundary[boundary[:, 1] < cols - margin * 2]
        else:
            chest_wall_x = 0 if centroid[1] < cols/2 else cols-1
            skin_points = boundary

        # 2. Sort Skin Points (Top-to-Bottom)
        if len(skin_points) == 0: return [], mask
        
        # Sort by row
        sort_idx = np.argsort(skin_points[:, 0])
        sorted_skin = skin_points[sort_idx]
        
        # 3. Subsample Start Points
        trim = int(len(sorted_skin) * 0.05)
        if trim > 0:
            active_skin = sorted_skin[trim:-trim]
        else:
            active_skin = sorted_skin
            
        indices = np.linspace(0, len(active_skin) - 1, num_paths, dtype=int)
        start_points = active_skin[indices]
        
        # Compute normals for these start points
        # Only compute normals for the selected points to save time/complexity
        # Use simple local tangent estimation
        normals = []
        smoothing = 10
        for idx in indices:
            # Map back to sorted_skin index
            # Need context for tangent
            idx_in_skin = idx # relative to active_skin? No, indices is into active_skin
            
            # Find neighbors in sorted_skin (which is continuous-ish)
            # Actually, sorted_skin might jump if contour is complex, but for breast skin it's usually monotonic y
            
            p_curr = active_skin[idx_in_skin]
            
            # Use points slightly up and down in the sorted list for robust tangent of the "macro" curve
            prev_idx = max(0, idx_in_skin - smoothing)
            next_idx = min(len(active_skin)-1, idx_in_skin + smoothing)
            
            p_prev = active_skin[prev_idx]
            p_next = active_skin[next_idx]
            
            tangent = p_next - p_prev
            if np.linalg.norm(tangent) == 0:
                normal = np.array([0, 1]) if chest_side == 'left' else np.array([0, -1])
            else:
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)
                
                # Check orientation: Should point towards chest wall x
                # Vector from start to chest wall
                to_wall = np.array([0, chest_wall_x - p_curr[1]])
                if np.dot(normal, to_wall) < 0:
                    normal = -normal
            normals.append(normal)
            
        
        paths = []
        for start, normal in zip(start_points, normals):
            # Start: Skin
            p0 = start
            
            # End: Chest wall (same row as start to ensure non-crossing macro behavior)
            p3 = np.array([start[0], chest_wall_x])
            
            dist = np.linalg.norm(p3 - p0)
            
            # Control Point 1: Extend along normal from skin
            # Length = fraction of distance to wall, e.g. 30%
            p1 = p0 + normal * (dist * 0.3)
            
            # Control Point 2: Guide towards the wall, maybe horizontally aligned with P3?
            # To ensure smooth landing at chest wall, P2 should be horizontally aligned with P3
            # So the curve arrives perpendicular to the wall (parallel to image rows)
            # Or just interpolate.
            
            # Let's align P2 with P3 to have 'horizontal' arrival at chest wall
            dir_to_wall = p3 - p0
            dir_to_wall /= (np.linalg.norm(dir_to_wall) + 1e-6)
            
            p2 = p3 - dir_to_wall * (dist * 0.3)
            # Or better: P2 is "in front of" P3. If P3 is (r, 0), P2 is (r, 50).
            # Vector pointing OUT from wall is -dir_to_wall basically.
            
            # Cubic Bezier
            path = self._cubic_bezier(p0, p1, p2, p3, num_points=100)
            
            # Valid mask check
            valid_mask = []
            for (r, c) in path:
                ri, ci = int(np.clip(r, 0, rows-1)), int(np.clip(c, 0, cols-1))
                valid_mask.append(mask[ri, ci])
            valid_mask = np.array(valid_mask)
            
            path_in_mask = path[valid_mask]
            
            if len(path_in_mask) > 10:
                paths.append(path_in_mask)
        
        return paths, mask

    def _cubic_bezier(self, p0, p1, p2, p3, num_points=100):
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

    def sample_paths_adaptively(self, paths, method='greedy', error_threshold=0.015, max_points=25, 
                                dense_step=100, sparse_step=300, gradient_window=50):
        """
        Samples points along paths.
        
        Args:
            method: 'greedy' (iterative error minimization) or 'gradient' (original heuristic).
            error_threshold: Max allowed interpolation error (normalized 0-1) for greedy method.
            max_points: Max points per path for greedy method.
        """
        if method == 'gradient':
            return self._sample_paths_gradient(paths, dense_step, sparse_step, gradient_window)
        else:
            return self._sample_paths_greedy(paths, error_threshold, max_points)

    def _sample_paths_greedy(self, paths, error_threshold=0.015, max_points=25):
        """
        Greedy sampling: Iteratively adds points where linear interpolation error is highest.
        """
        import scipy.ndimage
        sampled = []
        
        for path in paths:
            if len(path) < 2: continue
            
            # 1. Extract Full Profile at 1px-ish resolution
            diffs = np.diff(path, axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0], np.cumsum(lengths)])
            total_length = cumulative[-1]
            
            num_analysis = int(total_length)
            if num_analysis < 10: 
                sampled.append(path)
                continue
                
            analysis_t = np.linspace(0, total_length, num_analysis)
            
            # Interpolate coordinates
            # Vectorized interpolation
            # Note: Path points are not necessarily equidistant, so we map analysis_t to path segment indices
            analysis_indices = np.searchsorted(cumulative, analysis_t) - 1
            analysis_indices = np.clip(analysis_indices, 0, len(path) - 2)
            
            segment_t = (analysis_t - cumulative[analysis_indices]) / \
                        (cumulative[analysis_indices+1] - cumulative[analysis_indices] + 1e-6)
            
            r_coords = path[analysis_indices, 0] * (1 - segment_t) + path[analysis_indices+1, 0] * segment_t
            c_coords = path[analysis_indices, 1] * (1 - segment_t) + path[analysis_indices+1, 1] * segment_t
            
            coords = np.vstack((r_coords, c_coords))
            profile = scipy.ndimage.map_coordinates(self.blurred_pixel_array, coords, order=1, mode='nearest')
            
            # Normalize Profile (0-1) for consistent thresholding
            p_min, p_max = np.min(profile), np.max(profile)
            if p_max - p_min < 1e-6:
                sampled.append(np.array([path[0], path[-1]])) # Flat profile
                continue
                
            profile_norm = (profile - p_min) / (p_max - p_min)
            
            # 2. Iterative Selection
            # Start with endpoints
            indices = [0, num_analysis-1]
            
            # Add max gradient point (Heuristic initialization to catch the main edge fast)
            grad = np.abs(np.gradient(profile_norm))
            grad[0:10] = 0; grad[-10:] = 0 # Ignore boundary effects
            peak_idx = np.argmax(grad)
            if peak_idx not in indices: indices.append(peak_idx)
            
            indices.sort()
            
            for _ in range(max_points - len(indices)):
                # Interpolate from current set
                current_t = analysis_t[indices]
                current_vals = profile_norm[indices]
                
                interpolated = np.interp(analysis_t, current_t, current_vals)
                
                # Compute Error
                error = np.abs(profile_norm - interpolated)
                
                # Find Max Error
                max_err_idx = np.argmax(error)
                if error[max_err_idx] < error_threshold:
                    break
                    
                if max_err_idx not in indices:
                    indices.append(max_err_idx)
                    indices.sort()
            
            # Map indices back to coordinates
            final_r = r_coords[indices]
            final_c = c_coords[indices]
            final_points = np.vstack((final_r, final_c)).T
            
            sampled.append(final_points)
            
        return sampled

    def _sample_paths_gradient(self, paths, dense_step=100, sparse_step=300, gradient_window=50, search_limit_px=750, skip_skin_px=50):
        import scipy.ndimage
        
        sampled = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # 1. Create a dense temporary profile for analysis
            diffs = np.diff(path, axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0], np.cumsum(lengths)])
            total_length = cumulative[-1]
            
            # Resample the path at 5px resolution for analysis
            num_analysis_points = int(total_length / 5)
            if num_analysis_points < 10: 
                sampled.append(path) # Too short
                continue
                
            analysis_t = np.linspace(0, total_length, num_analysis_points)
            
            # Vectorized interpolation for analysis points
            analysis_indices = np.searchsorted(cumulative, analysis_t) - 1
            analysis_indices = np.clip(analysis_indices, 0, len(path) - 2)
            
            segment_t = (analysis_t - cumulative[analysis_indices]) / \
                        (cumulative[analysis_indices+1] - cumulative[analysis_indices] + 1e-6)
            
            r_coords = path[analysis_indices, 0] * (1 - segment_t) + path[analysis_indices+1, 0] * segment_t
            c_coords = path[analysis_indices, 1] * (1 - segment_t) + path[analysis_indices+1, 1] * segment_t
            
            # Extract intensity profile
            coords = np.vstack((r_coords, c_coords))
            # Interpolate the pixel values at analysis coordinates by picking the nearest pixel value to the coordinate value
            profile = scipy.ndimage.map_coordinates(self.blurred_pixel_array, coords, order=1, mode='nearest')
            
            # 2. Compute Gradient Magnitude (Robust)
            # Use heavy smoothing to find the "macro" artifact trend, ignoring local noise/lesions
            # profile_smooth = scipy.ndimage.gaussian_filter1d(profile, sigma=5) # Increased smoothing
            gradient = np.abs(np.gradient(profile))
            
            # 3. Find the "Event" (Tissue Thickness Change)
            # Skip skin entrance
            skip_skin_idx = int(skip_skin_px / 5)
            # Limit search
            search_limit_idx = int(search_limit_px / 5) 
            
            valid_gradient = gradient[skip_skin_idx:min(len(gradient), search_limit_idx)]
            
            if len(valid_gradient) > 0:
                peak_idx = np.argmax(valid_gradient) + skip_skin_idx
            else:
                peak_idx = 0
                
            # Define Dense Region: Peak +/- window
            window_indices = int(gradient_window / 5)
            dense_start_idx = max(0, peak_idx - window_indices)
            dense_end_idx = min(len(analysis_t) - 1, peak_idx + window_indices + int(100/5))
            
            dense_start_dist = analysis_t[dense_start_idx]
            dense_end_dist = analysis_t[dense_end_idx]
            
            # 4. Generate Final Sample Points
            target_lengths = [0.0]
            current_dist = 0.0
            
            while current_dist < total_length:
                # Check if we are in dense region
                is_dense = (current_dist >= dense_start_dist) and (current_dist <= dense_end_dist)
                
                step = dense_step if is_dense else sparse_step
                
                current_dist += step
                if current_dist <= total_length:
                    target_lengths.append(current_dist)
            
            target_lengths = np.array(target_lengths)
            
            # Interpolate final sample points
            final_indices = np.searchsorted(cumulative, target_lengths) - 1
            final_indices = np.clip(final_indices, 0, len(path) - 2)
            
            t_final = (target_lengths - cumulative[final_indices]) / \
                      (cumulative[final_indices+1] - cumulative[final_indices] + 1e-6)
            
            final_points = path[final_indices] * (1 - t_final)[:, np.newaxis] + \
                           path[final_indices+1] * t_final[:, np.newaxis]
                           
            sampled.append(final_points)
        
        return sampled

    def compute_correction_at_points(self, sample_points, mask, min_value=0.5, max_value=2.0):
        """
        Computes correction factors at sampled points based on local intensity.
        
        Uses median intensity from interior region as target.
        
        Args:
            sample_points: List of Mx2 arrays (one per path)
            mask: Binary mask of breast region
            
        Returns:
            tuple: (all_points, all_corrections) - flattened arrays
        """
        from scipy.ndimage import distance_transform_edt
        
        rows, cols = self.pixel_array.shape
        
        # Compute distance transform for target estimation
        distances = distance_transform_edt(mask)
        
        # Estimate target intensity from deep interior (high distance)
        max_dist = np.max(distances)
        interior_mask = distances > 0.0 * max_dist
        target_intensity = np.median(self.blurred_pixel_array[interior_mask])
        
        all_points = []
        all_corrections = []
        
        for points in sample_points:
            for (r, c) in points:
                ri = int(np.clip(r, 0, rows - 1))
                ci = int(np.clip(c, 0, cols - 1))
                
                if mask[ri, ci]:
                    # Local intensity (use small neighborhood for robustness)
                    r_min, r_max = max(0, ri-2), min(rows, ri+3)
                    c_min, c_max = max(0, ci-2), min(cols, ci+3)
                    local_intensity = np.median(self.blurred_pixel_array[r_min:r_max, c_min:c_max])
                    
                    # Correction factor
                    if local_intensity > 0:
                        correction = target_intensity / local_intensity
                        correction = np.clip(correction, min_value, max_value)
                    else:
                        correction = 1.0
                    
                    all_points.append([r, c])
                    all_corrections.append(correction)
        
        return np.array(all_points), np.array(all_corrections)

    def interpolate_correction_field(self, points, corrections, mask, method='cubic'):
        """
        Interpolates correction values to create 2D field.
        
        Args:
            points: Nx2 array of sample points
            corrections: N array of correction values
            mask: Binary mask
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            2D correction field
        """
        from scipy.interpolate import griddata
        
        rows, cols = self.pixel_array.shape
        
        # Create grid
        grid_r, grid_c = np.mgrid[0:rows, 0:cols]
        
        # Interpolate
        field = griddata(points, corrections, (grid_r, grid_c), 
                        method=method, fill_value=1.0)
        
        # Apply mask (correction is 1.0 outside breast)
        field[~mask] = 1.0
        
        # Handle any remaining NaNs
        field = np.nan_to_num(field, nan=1.0)
        
        return field

    def smooth_correction_field(self, field, sigma=10, min_value=0.5, max_value=2.0):
        """
        Applies Gaussian smoothing to the correction field.
        
        Args:
            field: 2D correction field
            sigma: Smoothing sigma
            
        Returns:
            Smoothed field
        """
        from scipy.ndimage import gaussian_filter
        
        smoothed = gaussian_filter(field, sigma=sigma)
        
        # Clip to reasonable range
        smoothed = np.clip(smoothed, min_value, max_value)
        
        return smoothed

    def _compute_rbf_matrix(self, source_points, target_points, kernel='thin_plate'):
        """
        Computes the interpolation matrix M such that M @ values = interpolated_values.
        Uses Radial Basis Functions (TPS) for global smoothness.
        
        Args:
            source_points: (N, 2) control points.
            target_points: (M, 2) target coordinates.
            kernel: 'thin_plate', 'cubic', 'linear', or 'gaussian'.
            
        Returns:
            np.ndarray: (M, N) dense matrix.
        """
        from scipy.spatial.distance import cdist
        
        # Kernel Functions
        def rbf_kernel(r, method='thin_plate', epsilon=1.0):
            if method == 'thin_plate':
                return np.where(r == 0, 0, r**2 * np.log(r + 1e-10))
            elif method == 'cubic':
                return r**3
            elif method == 'linear':
                return r
            elif method == 'gaussian':
                return np.exp(-(r/epsilon)**2)
            elif method == 'multiquadric':
                return np.sqrt((r/epsilon)**2 + 1)
            else:
                return r
                
        # 1. Compute Kernel Matrix for Source Points (K_cc)
        # Add regularization to diagonal for stability
        d_cc = cdist(source_points, source_points)
        K_cc = rbf_kernel(d_cc, method=kernel)
        np.fill_diagonal(K_cc, K_cc.diagonal() + 1e-8) # Regularization
        
        # 2. Compute Inverse of K_cc
        # Solve for interpolation weights: K_cc * w = values -> w = K_cc_inv * values
        # We want matrix M such that M * values = interpolated
        # Interpolated = K_tc * w = K_tc * (K_cc_inv * values) = (K_tc * K_cc_inv) * values
        # So M = K_tc @ K_cc_inv
        try:
            K_cc_inv = np.linalg.inv(K_cc)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (e.g. duplicate points)
            K_cc_inv = np.linalg.pinv(K_cc)
            
        # 3. Compute Kernel Matrix for Target Points (K_tc)
        d_tc = cdist(target_points, source_points)
        K_tc = rbf_kernel(d_tc, method=kernel)
        
        # 4. Compute Final Matrix M
        M = K_tc @ K_cc_inv
        
        return M

    def correct(self, num_paths=15, method='greedy', error_threshold=0.015, max_points=25, 
                dense_step=100, sparse_step=300, gradient_window=50,
                interpolation='cubic', smoothing_sigma=10):
        """
        High-level method to correct using path-based interpolation.
        
        Args:
            num_paths: Number of Bezier curves.
            method: 'greedy' or 'gradient'.
            error_threshold, max_points: Params for greedy.
            dense_step, sparse_step, gradient_window: Params for gradient.
        """
        # Generate paths
        paths, mask = self.generate_bezier_paths(num_paths=num_paths)
        
        if not paths:
            # Fallback: return uncorrected
            return self.copy(), np.ones_like(self.pixel_array), [], []
        
        # Sample points adaptively
        sample_points = self.sample_paths_adaptively(
            paths, method=method, error_threshold=error_threshold, max_points=max_points,
            dense_step=dense_step, sparse_step=sparse_step, gradient_window=gradient_window
        )
        
        # Compute corrections at sample points
        all_points, all_corrections = self.compute_correction_at_points(
            sample_points, mask
        )
        
        if len(all_points) < 3:
            return self.copy(), np.ones_like(self.pixel_array), paths, sample_points
        
        # Interpolate to 2D field
        field = self.interpolate_correction_field(
            all_points, all_corrections, mask, method=interpolation
        )
        
        # Smooth the field
        field = self.smooth_correction_field(field, sigma=smoothing_sigma)
        
        # Apply correction
        corrected_pixels = self.pixel_array * field
        corrected_image = PathCorrectedImage(pixel_array=corrected_pixels)
        
        return corrected_image, field, paths, sample_points

    def correct_optimized(self, num_paths=15, method='greedy', error_threshold=0.015, max_points=25, 
                          dense_step=100, sparse_step=300, gradient_window=50,
                          regularization_weight=1.0, downsample_factor=4, field_bounds=(0.1, 5.0)):
        """
        Corrects image by optimizing correction factors to minimize intensity standard deviation.
        
        Args:
            num_paths, ...: Sampling parameters (see correct method)
            smoothing_sigma: Sigma for final field smoothing
            regularization_weight: Weight for smoothness/drift penalty
            downsample_factor: Factor to downsample image for faster optimization (default 4)
            
        Returns:
            tuple: (corrected_image, correction_field, paths, sample_points)
        """
        from scipy.optimize import minimize
        from scipy.interpolate import griddata
        
        # 1. Generate geometry (paths and sample points)
        paths, mask = self.generate_bezier_paths(num_paths=num_paths)
        if not paths:
            return self.correct(num_paths=num_paths) # Fallback
            
        sample_points = self.sample_paths_adaptively(
            paths, method=method, error_threshold=error_threshold, max_points=max_points,
            dense_step=dense_step, sparse_step=sparse_step, gradient_window=gradient_window
        )
        
        # Initial guess: Use heuristic correction
        # Note: compute_correction_at_points filters out points outside mask! we must use ITS output points.
        print("Computing heuristic guess...")
        all_pts_h, heuristic_factors = self.compute_correction_at_points(sample_points, mask)
        
        all_points = all_pts_h
        n_points = len(all_points)
        initial_factors = heuristic_factors
        
        initial_factors = heuristic_factors
        
        # 2. Prepare data for optimization (Downsampled)
        if downsample_factor > 1:
            img_small = self.blurred_pixel_array[::downsample_factor, ::downsample_factor]
            mask_small = mask[::downsample_factor, ::downsample_factor]
            scale = 1.0 / downsample_factor
            opt_points = all_points * scale
        else:
            img_small = self.blurred_pixel_array
            mask_small = mask
            opt_points = all_points
            
        # Pre-compute masked image pixels for fast std calculation
        valid_pixels = img_small[mask_small]
        
        # -- PRE-COMPUTE INTERPOLATION MATRIX --
        # Find coordinates of all valid pixels
        r_coords, c_coords = np.where(mask_small)
        valid_coords = np.vstack((r_coords, c_coords)).T
        
        print(f"Computing RBF interpolation matrix for {len(valid_coords)} pixels and {len(opt_points)} control points...")
        # Use Thin Plate Spline for smooth global interpolation
        interp_matrix = self._compute_rbf_matrix(opt_points, valid_coords, kernel='thin_plate')
        
        # 3. Define Objective Function
        def objective(factors):
            # Fast Matrix Multiplication
            # field_values = M @ factors
            field_values = interp_matrix.dot(factors)
            
            # Apply correction
            corrected_pixels = valid_pixels * field_values
            
            # Metric: Standard Deviation
            std_dev = np.std(corrected_pixels)
            
            # Regularization: Penalize deviation from 1.0 (drift)
            reg_drift = np.mean((factors - 1.0)**2)
            
            # Total Loss
            return std_dev + regularization_weight * reg_drift
            
        # 4. Run Optimization
        n_points = len(initial_factors)
        bounds = [field_bounds for _ in range(n_points)]
        
        print(f"Optimizing {n_points} variables on {len(valid_pixels)} pixels...")
        result = minimize(objective, initial_factors, bounds=bounds, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-4})
                          
        optimal_factors = result.x
        loss = result.fun
        print(f"Optimization finished: {result.message} (Iterations: {result.nit})")
        
        # 5. Apply Final Correction (Full Resolution)
        # Use cubic interpolation for the final high-quality result
        field = self.interpolate_correction_field(
            all_points, optimal_factors, mask, method='cubic'
        )
        
        # Create the corrected image and force the intensity to be equal to the original image
        corrected_pixels = self.pixel_array * field
        intensity = np.sum(self.pixel_array)/np.sum(corrected_pixels)
        corrected_pixels = intensity*corrected_pixels
        corrected_image = PathCorrectedImage(pixel_array=corrected_pixels)
        
        return corrected_image, field, paths, sample_points, loss

    def visualize_paths(self, paths=None, sample_points=None, ax=None):
        """
        Visualizes the paths and sample points on the image.
        
        Args:
            paths: List of paths (if None, generates new ones)
            sample_points: List of sample points (if None, samples from paths)
            ax: Matplotlib axis (if None, creates new figure)
            
        Returns:
            matplotlib axis
        """
        import matplotlib.pyplot as plt
        
        if paths is None:
            paths, _ = self.generate_bezier_paths()
        
        if sample_points is None and paths:
            sample_points = self.sample_paths_adaptively(paths)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show image
        ax.imshow(self.pixel_array, cmap='gray')
        
        # Plot paths
        for path in paths:
            ax.plot(path[:, 1], path[:, 0], 'y-', linewidth=1.5, alpha=0.8)
        
        # Plot sample points
        for points in sample_points:
            ax.scatter(points[:, 1], points[:, 0], c='red', s=30, zorder=5)
        
        ax.set_title('Generated Non-Crossing Correction Paths')
        ax.axis('off')
        
        return ax

    def visualize(self, save=False, figname='bib_correction_result.png', **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Extract args relevant for heuristic correction (subset of kwargs)
        valid_correct_args = [
            'num_paths', 'method', 'error_threshold', 'max_points', 
            'dense_step', 'sparse_step', 'gradient_window', 
            'interpolation', 'smoothing_sigma'
        ]
        correct_kwargs = {k: v for k, v in kwargs.items() if k in valid_correct_args}
        
        # Ensure consistent num_paths used everywhere if not provided
        num_paths = kwargs.get('num_paths', 12) # Original visualize default
        if 'num_paths' not in correct_kwargs:
            correct_kwargs['num_paths'] = num_paths

        # 1. Heuristic Correction
        corr_h, field_h, _, _ = self.correct(**correct_kwargs)

        mask, _ = self.detect_boundary()
        std_orig = np.std(self.pixel_array[mask])
        std_h = np.std(corr_h.pixel_array[mask])
        red_h = (std_orig - std_h) / std_orig * 100

        print(f"Heuristic Std: {std_h:.3f} (Reduction: {red_h:.1f}%)")

        # 2. Optimization Correction
        print("\n[2] Running Optimization Correction...")
        
        # Ensure num_paths passed to optimization if not in kwargs
        opt_kwargs = kwargs.copy()
        if 'num_paths' not in opt_kwargs:
            opt_kwargs['num_paths'] = num_paths
            
        corr_opt, field_opt, _, _, _ = self.correct_optimized(**opt_kwargs)

        std_opt = np.std(corr_opt.pixel_array[mask])
        print(f"Original shape: {std_orig.shape}")
        print(f"Optimized shape: {std_opt.shape}")
        red_opt = (std_orig - std_opt) / std_orig * 100

        print(f"Optimized Std: {std_opt:.3f} (Reduction: {red_opt:.1f}%)")

        # 3. Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Images
        vmin, vmax = np.percentile(self.pixel_array, (1, 99))

        axes[0, 0].imshow(self.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Original (Std: {std_orig:.0f})")

        axes[0, 1].imshow(corr_h.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Heuristic (Red: {red_h:.1f}%)")

        axes[0, 2].imshow(corr_opt.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f"Optimized (Red: {red_opt:.1f}%)")

        # Row 2: Fields
        # Re-generate visuals matching the parameters
        paths, _ = self.generate_bezier_paths(num_paths)
        
        # Extract sampling args for visualizaton
        sampling_args = ['method', 'error_threshold', 'max_points', 'dense_step', 'sparse_step', 'gradient_window']
        samp_kwargs = {k: v for k, v in kwargs.items() if k in sampling_args}
        
        sample_points = self.sample_paths_adaptively(paths, **samp_kwargs)
        self.visualize_paths(paths, sample_points, axes[1, 0])

        vmin = np.min([np.min(field_h), np.min(field_opt)])
        vmax = np.max([np.max(field_h), np.max(field_opt)])

        divider = make_axes_locatable(axes[1,2])
        cax = divider.append_axes('right', size='5%', pad=0.05)

        axes[1, 1].imshow(field_h, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("Heuristic Field")

        field_opt_im = axes[1, 2].imshow(field_opt, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(field_opt_im, cax=cax)
        axes[1, 2].set_title("Optimized Field")

        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        if save:
            plt.savefig(figname, dpi=150)
        else:
            return axes
        
    def _normalized_gaussian(self, sigma, radius):
        """
        Applies Normalized Convolution to ignore background zeros during smoothing.
        
        Strategy:
        1. Shift image intensities to be strictly negative (val - peak).
           This ensures that '0' can be uniquely used as the 'missing data' value
           for the padding/background, distinct from dark tissue (which is negative).
        2. Apply mask (background -> 0).
        3. Blur signal and mask.
        4. Normalize (Divide).
        5. Shift back (+ peak).
        """
        import scipy.ndimage
        
        mask, _ = self.detect_boundary()
        mask_float = mask.astype(float)
        
        peak = np.max(self.pixel_array)
        shifted_image = self.pixel_array - peak 
        
        image_masked = shifted_image * mask_float
        
        blurred_image = scipy.ndimage.gaussian_filter(image_masked, sigma=sigma, radius=radius)
        blurred_mask = scipy.ndimage.gaussian_filter(mask_float, sigma=sigma, radius=radius)
        
        result = np.zeros_like(blurred_image)
        # Avoid division by zero
        valid_indices = blurred_mask > 1e-4
        result[valid_indices] = blurred_image[valid_indices] / blurred_mask[valid_indices]
        
        restored = result + peak
        
        sum_restored = np.sum(restored)
        if sum_restored == 0:
            return restored
        
        scale_factor = np.sum(self.pixel_array) / sum_restored
        
        # Enforce original boundary (fix "blooming" effect)
        final_result = restored * scale_factor * mask_float 
        
        return final_result

