"""
Breast artifact correction using (s, d) curvilinear coordinates.

This module defines the SDCorrectedImage class, which fits a 2D B-spline
surface in arc-length (s) and depth (d) coordinates to correct halo
artifacts in CEM images.

Coordinates:
  d: Normalized distance from the skin boundary [0, 1].
  s: Normalized arc-length along the contour [0, 1].
"""

import numpy as np
import scipy.ndimage

try:
    from .path_correction import PathCorrectedImage
except ImportError:
    from path_correction import PathCorrectedImage


class SDCorrectedImage(PathCorrectedImage):
    """
    CEM image with (s, d) curvilinear coordinate halo-artifact correction.

    Adds methods to correct artifacts using (s, d) coordinate mapping:
        correct_sd(): Fast heuristic profile correction.
        correct_sd_optimized(): L-BFGS-B optimization of the spline grid.

    Parameters
    ----------
    dicom_path : str, optional
        Path to DICOM file.
    pixel_array : np.ndarray, optional
        2-D array of pixel values.
    sigma, radius : float, int
        Parameters for normalized convolution blurring (default 25, 50).
    """

    def __init__(self, dicom_path=None, pixel_array=None, sigma=25, radius=50):
        super().__init__(dicom_path=dicom_path, pixel_array=pixel_array,
                            sigma=sigma, radius=radius)
        self._sd_mask = None
        self._sd_boundary = None
        self._sd_skin_ring = None
        self._sd_edge_ring = None
        self._sd_on_edge = None
        self._sd_d_map = None
        self._sd_d_max = None
        self._sd_s_map = None
        self._sd_s_total = None
        self._sd_d_norm = None
        self._sd_s_norm = None
        self._sd_shadow_mask = None
        self._sd_s_origin = None
        self._sd_s_end = None
        self._last_field = None

    def _build_sd_maps(
            self,
            d_max_percentile=97,
            mask_to_boundary=3,
            **kwargs
    ):
        """
        Builds the s-d-coordinate system.

        Parameters
        ----------
        d_max_percentile : int
            Percentile used for depth normalization (default 97).
        mask_to_boundary : int
            Pulls the mask mask_to_boundary pixels towards the image boundaries.

        Populates:
            _sd_norm_
            _sd_maps_params
            _sd_s_total
            _sd_s_origin
            _sd_s_end
            _sd_d_map
            _sd_d_max
            _sd_d_norm
            _sd_s_norm
        """
        from scipy.ndimage import distance_transform_edt, binary_erosion, gaussian_filter
        from scipy.spatial import KDTree
        from skimage.measure import find_contours

        # Map caching
        current_params = {
            'd_max_percentile':    d_max_percentile,
            **kwargs
        }
        if (self._sd_d_norm is not None and
                getattr(self, '_sd_maps_params', None) == current_params):
            return
        self._sd_maps_params = current_params

        # Create mask
        mask, _ = self.detect_boundary(threshold=kwargs.get('threshold'))
        mask[:mask_to_boundary,:] = np.repeat(np.any(mask[:mask_to_boundary,:], axis=0)[np.newaxis,:], mask_to_boundary, axis=0)
        mask[-mask_to_boundary:,:] = np.repeat(np.any(mask[-mask_to_boundary:,:], axis=0)[np.newaxis,:], mask_to_boundary, axis=0)
        mask[:,:mask_to_boundary] = np.repeat(np.any(mask[:,:mask_to_boundary], axis=1)[:,np.newaxis], mask_to_boundary, axis=1)
        mask[:,-mask_to_boundary:] = np.repeat(np.any(mask[:,-mask_to_boundary:], axis=1)[:,np.newaxis], mask_to_boundary, axis=1)
        self._sd_mask = mask

        # Boundary contour
        contours = find_contours(mask.astype(float), 0.99)
        if not contours:
            raise ValueError("No breast contour found — cannot build (s,d) maps.")
        boundary = max(contours, key=len)   # (N, 2) in (row, col)
        self._sd_boundary = boundary

        # Set 0 and 1 coordinates for s
        s0 = boundary[-1,:]; s1 = boundary[0,:]

        # Scale arc-lengths to [0, 1] across the skin-wall span
        seg_lengths = np.linalg.norm(np.diff(boundary, axis=0), axis=1)
        arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        skin_start = float(arc_lengths[0])
        skin_span = float(arc_lengths[-1] - arc_lengths[0])
        skin_arc_lengths = arc_lengths - skin_start

        self._sd_s_total = skin_arc_lengths[-1].astype(float)
        self._sd_s_origin = s0   # image-frame point, s = 0
        self._sd_s_end = s1   # image-frame point, s = total_span

        boundary_mask = np.full(mask.shape, False)
        br = boundary.astype(int)[:,0]; bc = boundary.astype(int)[:,1]
        boundary_mask[br,bc] = True

        # Compute d-coordinates
        d_map_full, nearest_idx = distance_transform_edt(~boundary_mask, return_indices=True)
        self._sd_d_map = np.where(mask, d_map_full, 0.0)
        self._sd_d_max = float(np.percentile(self._sd_d_map[mask], d_max_percentile))
        

        # Assign arc-lengths to every skin-ring pixel
        ctree = scipy.spatial.KDTree(boundary)
        rr, cc = np.where(boundary_mask)
        _, nn  = ctree.query(np.column_stack([rr, cc]))
        arc_length_map = np.zeros(mask.shape, dtype=np.float64)
        arc_length_map[rr, cc] = skin_arc_lengths[nn]

        # Propagate to interior pixels via nearest-skin-pixel indices
        s_map_raw = np.where(
            mask,
            arc_length_map[nearest_idx[0], nearest_idx[1]],
            0.0
        )

        self._sd_s_map = s_map_raw
        
        # Normalised depth
        self._sd_d_norm = np.clip(self._sd_d_map / self._sd_d_max, 0.0, 1.0)

        self._sd_s_norm = np.where(mask, self._sd_s_map / self._sd_s_total, 0.)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_heuristic_grid(self, sc, dc, ns, nd,
                              field_bounds=(0.2, 3.0),
                              d_ref_threshold=0.70,
                              **kwargs):
        """
        Builds the initial correction grid from the binned mean intensity profile.

        Parameters
        ----------
        sc, dc : arrays
            Bin centers for s and d coordinates.
        ns, nd : int
            Grid dimensions along s and d.
        field_bounds : (float, float)
            Min/max clip range for the grid (default (0.2, 3.0)).
        d_ref_threshold : float
            Depth threshold for defining reference tissue (default 0.70).

        Returns
        -------
        bin_grid : (ns, nd) array
            Initial correction factor estimates.
        """
        from scipy.stats import binned_statistic_2d

        # Strategy: bin pixel intensities by (s, d) and estimate target from deep tissue
        mask     = self._sd_mask
        s_norm   = self._sd_s_norm
        d_norm   = self._sd_d_norm

        bin_cem, _, _, _ = binned_statistic_2d(
            s_norm[mask], d_norm[mask], self.pixel_array[mask],
            statistic='mean', bins=[ns, nd], range=[[0, 1], [0, 1]]
        )
        bin_cem = np.where(np.isnan(bin_cem), np.nanmedian(bin_cem), bin_cem)

        # Reference intensity: mean of deep-interior bins
        target = float(np.nanmean(bin_cem[:, dc > d_ref_threshold]))

        bin_grid = np.clip(
            np.where(bin_cem > 1e-3, target / bin_cem, 1.0),
            field_bounds[0], field_bounds[1]
        )
        return bin_grid
    
    def _compute_adaptive_knots(self, ns, nd,
                s_margin=0.01,
                d_margin=0.005,
                d_strategy='gradient',
                d_beta_a=0.6,
                d_beta_b=2.5,
                knot_smooth_sigma=5.0,
                knot_weight_floor=0.05,
                min_strip_pixels=10,
                n_d_probe=100,
                **kwargs):
            """
            Sizes the spline grid (knots) adaptively based on the breast shape.

            S-knots are placed at quantiles, while d-knots can use different 
            strategies (gradient, curvature, etc.).

            Parameters
            ----------
            ns, nd : int
                Number of knots along s and d.
            s_margin, d_margin : float
                Percentile fractions to clip ranges (default 0.01, 0.005).
            d_strategy : str
                Placement logic: 'quantile', 'gradient', 'curvature', or 'beta'.
            d_beta_a, d_beta_b : float
                Beta distribution parameters for 'beta' strategy.
            knot_smooth_sigma : float
                Smoothing for profile differentiation (default 5.0).
            knot_weight_floor : float
                Min weight for coverage (default 0.05).
            min_strip_pixels : int
                Required pixels per shell for safe depth check (default 10).
            n_d_probe : int
                Number of depth levels probed for safe depth (default 100).

            Returns
            -------
            sc, dc : arrays
                Adaptive knot centers in s and d.
            """
            from scipy.stats import beta as beta_dist
            from scipy.ndimage import gaussian_filter1d

            self._build_sd_maps(**kwargs)
            mask   = self._sd_mask
            s_flat = self._sd_s_norm[mask].astype(np.float64)
            d_flat = self._sd_d_norm[mask].astype(np.float64)

            # s-knots: quantile spacing for equal coverage
            s_lo = float(np.percentile(s_flat, 100.0 * s_margin))
            s_hi = float(np.percentile(s_flat, 100.0 * (1.0 - s_margin)))
            q_lo_s = float(np.mean(s_flat <= s_lo))
            q_hi_s = float(np.mean(s_flat <= s_hi))
            sc = np.quantile(s_flat, np.linspace(q_lo_s, q_hi_s, ns))
            sc[0] = s_lo;  sc[-1] = s_hi
            # sc = np.maximum.accumulate(sc)
            # sc = np.minimum.accumulate(sc[::-1])[::-1]
            if not np.all(np.diff(sc) > 0):
                eps = np.finfo(float).eps * (s_hi - s_lo) * nd
                for i in range(1, len(sc)):
                    if sc[i] <= sc[i - 1]:
                        sc[i] = sc[i - 1] + eps
                sc[-1] = s_hi  # restore the hard upper bound

            # d-knots: placement by strategy
            d_lo = float(np.percentile(d_flat, 100.0 * d_margin))
            d_hi = float(np.percentile(d_flat, 100.0 * (1.0 - d_margin)))

            if d_strategy == 'quantile':
                q_lo_d = float(np.mean(d_flat <= d_lo))
                q_hi_d = float(np.mean(d_flat <= d_hi))
                dc = np.quantile(d_flat, np.linspace(q_lo_d, q_hi_d, nd))

            elif d_strategy in ('gradient', 'curvature'):
                n_prof  = 200
                d_grid  = np.linspace(d_lo, d_hi, n_prof)
                d_edges = np.concatenate([
                    [d_lo - 1e-9],
                    0.5 * (d_grid[:-1] + d_grid[1:]),
                    [d_hi + 1e-9],
                ])

                if hasattr(self, '_last_field') and self._last_field is not None:
                    f_flat = self._last_field[mask].astype(np.float64)
                else:
                    v      = self.pixel_array[mask].astype(np.float64)
                    v_med  = np.median(v)
                    f_flat = v_med / (v + 1e-8)

                f_profile = np.array([
                    f_flat[(d_flat >= d_edges[b]) & (d_flat < d_edges[b + 1])].mean()
                    if ((d_flat >= d_edges[b]) & (d_flat < d_edges[b + 1])).any()
                    else np.nan
                    for b in range(n_prof)
                ])

                nans = np.isnan(f_profile)
                if nans.any() and (~nans).sum() >= 2:
                    f_profile[nans] = np.interp(
                        d_grid[nans], d_grid[~nans], f_profile[~nans]
                    )

                f_smooth = gaussian_filter1d(f_profile, sigma=knot_smooth_sigma)

                if d_strategy == 'gradient':
                    weight = np.abs(np.gradient(f_smooth, d_grid))
                else:
                    weight = np.abs(
                        np.gradient(np.gradient(f_smooth, d_grid), d_grid)
                    )

                weight = weight + knot_weight_floor * weight.max()
                weight = np.maximum(weight, 0.0)

                cdf         = np.cumsum(weight)
                cdf         = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + 1e-12)
                dc          = np.interp(np.linspace(0.0, 1.0, nd), cdf, d_grid)

            elif d_strategy == 'beta':
                uniform = np.linspace(0.0, 1.0, nd)
                warped  = beta_dist.ppf(uniform, d_beta_a, d_beta_b)
                dc      = d_lo + warped * (d_hi - d_lo)

            else:
                raise ValueError(
                    f"d_strategy must be 'quantile', 'gradient', 'curvature', or "
                    f"'beta'; got '{d_strategy}'."
                )

            # Prevent knot overlap near the medial axis
            shell_half = 0.5 / n_d_probe
            d_probe    = np.linspace(d_lo, d_hi, n_d_probe)

            # Assign every masked pixel to its nearest s-column
            col_assign = np.argmin(
                np.abs(s_flat[:, None] - sc[None, :]), axis=1
            )

            safe_d_hi = d_hi
            for d_level in reversed(d_probe):
                in_shell = (
                    (d_flat >= d_level - shell_half) &
                    (d_flat <= d_level + shell_half)
                )
                all_ok = all(
                    int(np.sum(in_shell & (col_assign == i))) >= min_strip_pixels
                    for i in range(ns)
                )
                if all_ok:
                    safe_d_hi = float(d_level)
                    break

            d_hi = min(d_hi, safe_d_hi)

            # Enforce strict monotonicity
            # dc[0]  = d_lo;  dc[-1]  = d_hi
            # dc = np.maximum.accumulate(dc)
            # dc = np.minimum.accumulate(dc[::-1])[::-1]
            if not np.all(np.diff(dc) > 0):
                eps = np.finfo(float).eps * (d_hi - d_lo) * nd
                for i in range(1, len(dc)):
                    if dc[i] <= dc[i - 1]:
                        dc[i] = dc[i - 1] + eps
                dc[-1] = d_hi  # restore the hard upper bound

            return sc, dc

    def _build_interp_matrix(self, sc, dc, s_opt, d_opt, ns, nd):
        """
        Builds the interpolation matrix relating parameters to field values.
        """
        from scipy.interpolate import RectBivariateSpline

        n_par   = ns * nd
        n_pix   = len(s_opt)
        M       = np.zeros((n_pix, n_par), dtype=np.float64)
        impulse = np.zeros((ns, nd),       dtype=np.float64)

        for k in range(n_par):
            ki, kj          = divmod(k, nd)
            impulse[ki, kj] = 1.0
            M[:, k]         = RectBivariateSpline(
                sc, dc, impulse, kx=3, ky=3
            ).ev(s_opt, d_opt)
            impulse[ki, kj] = 0.0

        bc_idx        = np.array([], dtype=int)
        free_idx      = np.arange(n_par)
        fixed_contrib = np.zeros(n_pix, dtype=np.float64)

        fixed_contrib = M[:, bc_idx] @ np.ones(len(bc_idx), dtype=np.float64)

        return M[:, free_idx], free_idx, fixed_contrib, bc_idx

    def _compute_knot_positions(self, sc, dc, opt_r, opt_c, s_opt, d_opt):
        """
        Estimates the pixel coordinates of each spline knot.

        Uses a strip-based approach to follow iso-d rays inside the breast,
        ensuring knots are placed correctly relative to the tissue.

        Returns
        -------
        knot_xy : ndarray, shape (ns, nd, 2)  — knot_xy[i,j] = [mean_row, mean_col]
        """
        ns, nd = len(sc), len(dc)
        se = np.concatenate([[0.0], 0.5 * (sc[:-1] + sc[1:]), [1.0]])

        k_near   = 20
        knot_xy  = np.zeros((ns, nd, 2), dtype=np.float64)
        for i in range(ns):
            in_strip = (s_opt >= se[i]) & (s_opt < se[i + 1])
            if not in_strip.any():
                in_strip = np.ones(len(s_opt), dtype=bool)
            strip_r = opt_r[in_strip]
            strip_c = opt_c[in_strip]
            strip_d = d_opt[in_strip]
            for j in range(nd):
                k = min(k_near, len(strip_r))
                order = np.argsort(np.abs(strip_d - dc[j]))[:k]
                knot_xy[i, j, 0] = strip_r[order].mean()
                knot_xy[i, j, 1] = strip_c[order].mean()

        return knot_xy

    def _apply_medial_boundary_conditions(self, knot_xy, dc, M_free, free_idx,
                                          n_par, ns, nd,
                                          d_bc_thresh, bc_radius_px):
        """
        Forces coincident deep knots to share the same value (merged).

        This ensures smooth transitions in deep tissue where coordinate shells 
        might otherwise overlap.

        Returns
        -------
        M_free_bc   : reduced interpolation matrix (n_pix, n_free_bc)
        free_idx_bc : reduced free-parameter index array (n_free_bc,)
        merge_map   : dict {follower_flat_idx: leader_flat_idx}
                      After optimization: opt_par[follower] = opt_par[leader]
        """
        M_work    = M_free.copy()
        flat_to_k = {int(f): k for k, f in enumerate(free_idx)}
        merge_map = {}
        remove_k  = set()

        for j in range(nd):
            if float(dc[j]) < d_bc_thresh:
                continue

            knots_j = []
            for i in range(ns):
                flat = int(i * nd + j)
                if flat in flat_to_k:
                    knots_j.append((flat, knot_xy[i, j]))

            if len(knots_j) < 2:
                continue

            claimed = [False] * len(knots_j)
            for a in range(len(knots_j)):
                if claimed[a]:
                    continue
                flat_a, pos_a = knots_j[a]
                k_a = flat_to_k[flat_a]
                for b in range(a + 1, len(knots_j)):
                    if claimed[b]:
                        continue
                    flat_b, pos_b = knots_j[b]
                    if np.linalg.norm(pos_a - pos_b) <= bc_radius_px:
                        k_b = flat_to_k[flat_b]
                        M_work[:, k_a] += M_work[:, k_b]
                        merge_map[flat_b] = flat_a
                        remove_k.add(k_b)
                        claimed[b] = True

        keep = np.ones(len(free_idx), dtype=bool)
        for k in remove_k:
            keep[k] = False
        M_free_bc   = M_work[:, keep]
        free_idx_bc = free_idx[keep]

        n_merged = int(keep.size - keep.sum())
        if n_merged:
            print(f"  Medial BC: merged {n_merged} follower knot(s) → "
                  f"{len(free_idx_bc)} free params (was {len(free_idx)})", flush=True)
        else:
            print(f"  Medial BC: no knots within {bc_radius_px:.0f}px at "
                  f"d >= {d_bc_thresh:.2f} — try increasing bc_radius_px.", flush=True)

        return M_free_bc, free_idx_bc, merge_map, keep

    def _reg_and_grad(self, q, free_idx, ns, nd, lam_smooth, lam_drift):
        """
        Computes the regularization loss and its gradient for spline grid parameters.

        Both smoothness and drift are now computed in q-space (log-space):
        - Smoothness: penalizes differences in log(p) between adjacent knots,
                        i.e. multiplicative ratios p_ij / p_i'j'.
        - Drift:      penalizes q away from 0, equivalent to a log-normal prior
                        on p centered at 1 (neutral correction).

        Both terms have constant Hessians in q-space ((2λ_s/K)·L and (2λ_d/K_f)·I),
        fully consistent with the log reparametrization. The alpha parameter is kept
        in the signature for compatibility but is not used.
        """
        n_par  = ns * nd
        n_free = len(free_idx)

        # Reconstruct full q grid (fixed knots sit at q=0, i.e. p=1)
        q_full = np.zeros(n_par, dtype=np.float64)
        q_full[free_idx] = q
        q2d = q_full.reshape(ns, nd)

        # ── Smoothness in q-space: R_smooth = (λ_s/K) · q^T L q ─────────────────
        # Hessian: (2λ_s/K) · L  [constant, PSD]
        ds_ = np.diff(q2d, axis=0)
        dd_ = np.diff(q2d, axis=1)
        smooth_loss = lam_smooth * (np.sum(ds_**2) + np.sum(dd_**2)) / n_par

        dq = np.zeros((ns, nd), dtype=np.float64)
        dq[1:,  :] += 2 * ds_;  dq[:-1, :] -= 2 * ds_
        dq[:,  1:] += 2 * dd_;  dq[:,  :-1] -= 2 * dd_
        g_smooth_q = dq.ravel()[free_idx] * lam_smooth / n_par

        # ── Drift in q-space: R_drift = (λ_d/K_f) · ||q||^2 ─────────────────────
        # Hessian: (2λ_d/K_f) · I  [constant, PD]
        drift_loss = lam_drift * np.mean(q**2)
        g_drift_q  = 2.0 * lam_drift * q / n_free

        grad_q = g_smooth_q + g_drift_q
        return float(smooth_loss + drift_loss), grad_q


    def _hessp_std(self, q, vec, M_free, fixed_contrib, v_opt,
                free_idx, ns, nd, lam_smooth, lam_drift):
        """
        Hessian-vector product for Trust-Region Newton-CG.

        Data term:         computed in p-space (Gauss-Newton), converted to q-space
                        via the standard chain rule H^q = p ⊙ (H^p (p ⊙ v)) + ∇_p·p·v.
        Smoothness term:   H^q_smooth = (2λ_s/K)·L, constant in q-space, applied
                        directly without chain-rule conversion.
        Drift term:        H^q_drift  = (2λ_d/K_f)·I, constant in q-space, applied
                        directly without chain-rule conversion.
        """
        fp    = np.exp(q)
        fv    = M_free @ fp + fixed_contrib
        n_par = ns * nd
        n_free = len(free_idx)

        mu_v    = v_opt.mean()
        mu_vf   = (v_opt * fv).mean()
        alpha   = mu_v / (mu_vf + 1e-12)
        fv_norm = alpha * fv
        u       = v_opt * fv_norm
        mu_u    = u.mean()
        N       = len(u)
        sigma   = np.std(u) + 1e-12

        g_data      = v_opt * (u - mu_u) / (N * sigma)
        grad_data_p = alpha * (
            M_free.T @ g_data
            - (np.dot(fv_norm, g_data) / (N * mu_v)) * (M_free.T @ v_opt)
        )

        # p-space direction: p ⊙ vec
        p_vec = fp * vec

        # Data Hessian-vector product (Gauss-Newton, p-space)
        Mv     = M_free @ p_vec
        term1  = (alpha**2 / (N * sigma))     * (M_free.T @ (v_opt**2 * Mv))
        term2  = -(alpha**2 / (N**2 * sigma)) * (M_free.T @ v_opt) * (v_opt @ Mv)
        term3  = -(alpha**2 / (N * sigma**2)) * (M_free.T @ (v_opt * (u - mu_u))) * (g_data @ Mv)
        H_data_p_pvec = term1 + term2 + term3

        # Convert data term to q-space via chain rule:
        #   H^q z = p ⊙ (H^p (p ⊙ z)) + ∇_p L ⊙ p ⊙ z
        H_data_q = fp * H_data_p_pvec + grad_data_p * fp * vec

        # Smooth Hessian-vector product (H^q_smooth = (2λ_s/K)·L, constant)
        # Apply discrete 2-D Laplacian directly to vec (already in q-space).
        v_full = np.zeros(n_par, dtype=np.float64)
        v_full[free_idx] = vec
        v_grid = v_full.reshape(ns, nd)
        dvs = np.diff(v_grid, axis=0)
        dvd = np.diff(v_grid, axis=1)
        dv  = np.zeros((ns, nd), dtype=np.float64)
        dv[1:,  :] += 2 * dvs;  dv[:-1, :] -= 2 * dvs
        dv[:,  1:] += 2 * dvd;  dv[:,  :-1] -= 2 * dvd
        H_smooth_q = dv.ravel()[free_idx] * lam_smooth / n_par

        # Drift Hessian-vector product (H^q_drift = (2λ_d/K_f)·I, constant)
        H_drift_q = 2.0 * lam_drift / n_free * vec

        return (H_data_q + H_smooth_q + H_drift_q).astype(np.float64)


    def _objective_std(self, q, M_free, fixed_contrib, v_opt,
                    free_idx, ns, nd, lam_smooth, lam_drift):
        """
        Standard deviation objective function for L-BFGS optimization.
        """
        fp      = np.exp(q)
        fv      = M_free @ fp + fixed_contrib

        mu_v    = v_opt.mean()
        mu_vf   = (v_opt * fv).mean()
        alpha   = mu_v / (mu_vf + 1e-12)
        fv_norm = alpha * fv

        corr  = v_opt * fv_norm
        mu    = corr.mean()
        nm    = len(corr)
        sigma = np.std(corr)

        g           = v_opt * (corr - mu) / (nm * sigma + 1e-12)
        grad_data_p = alpha * (
            M_free.T @ g
            - (np.dot(fv_norm, g) / (nm * mu_v)) * (M_free.T @ v_opt)
        )
        grad_data_q = grad_data_p * fp

        reg_loss, g_reg_q = self._reg_and_grad(
            q, free_idx, ns, nd, lam_smooth, lam_drift
        )
        return float(sigma) + reg_loss, (grad_data_q + g_reg_q).astype(np.float64)


    def _objective_entropy(self, q, M_free, fixed_contrib, v_opt, g_opt,
                       free_idx, ns, nd, lam_smooth, lam_drift,
                       depth_weight, n_bins, hist_smooth_sigma,
                       adaptive_bandwidth=True):
        """
        Entropy-based objective (joint intensity-gradient distribution).

        Parameters
        ----------
        hist_smooth_sigma : float
            Minimum kernel bandwidth floor (default 1.5).
        adaptive_bandwidth : bool
            Use Silverman's rule to adjust bandwidth based on sample density.
        """
        from scipy.ndimage import gaussian_filter

        fp = np.exp(q)
        fv = M_free @ fp + fixed_contrib
        W  = depth_weight

        I_vals = v_opt * fv
        G_vals = g_opt * fv

        I_min, I_max = I_vals.min(), I_vals.max()
        G_min, G_max = G_vals.min(), G_vals.max()

        def _norm(v, vmin, vmax):
            return np.clip((v - vmin) / (vmax - vmin + 1e-8) * (n_bins - 1), 0, n_bins - 1)

        In = _norm(I_vals, I_min, I_max)
        Gn = _norm(G_vals, G_min, G_max)

        I_lo = np.floor(In).astype(int);  I_hi = np.minimum(I_lo + 1, n_bins - 1)
        G_lo = np.floor(Gn).astype(int);  G_hi = np.minimum(G_lo + 1, n_bins - 1)
        wi   = In - I_lo
        wg   = Gn - G_lo

        H2d = np.zeros((n_bins, n_bins), dtype=np.float64)
        np.add.at(H2d, (I_lo, G_lo), W * (1 - wi) * (1 - wg))
        np.add.at(H2d, (I_hi, G_lo), W * wi*(1 - wg))
        np.add.at(H2d, (I_lo, G_hi), W * (1 - wi)*wg)
        np.add.at(H2d, (I_hi, G_hi), W * wi*wg)

        # Bandwidth selection
        if adaptive_bandwidth:
            sum_W  = float(W.sum())
            sum_W2 = float((W**2).sum())
            n_eff  = (sum_W**2) / (sum_W2 + 1e-12)
            sigma_I = max(hist_smooth_sigma, 1.06 * float(np.std(In)) * n_eff**(-0.2))
            sigma_G = max(hist_smooth_sigma, 1.06 * float(np.std(Gn)) * n_eff**(-0.2))
        else:
            sigma_I = hist_smooth_sigma
            sigma_G = hist_smooth_sigma

        # Smoothed histogram and entropy calculation
        H2d_smooth = gaussian_filter(H2d, sigma=[sigma_I, sigma_G]) + 1e-10
        P          = H2d_smooth / H2d_smooth.sum()
        H_ent      = float(-np.sum(P * np.log(P)))

        # Gradient back-prop through Gaussian filter
        dH_dH2d_smooth = -(1.0 + np.log(P)) / H2d_smooth.sum()
        dH_dH2d_raw    = gaussian_filter(dH_dH2d_smooth, sigma=[sigma_I, sigma_G])

        scale_I = (n_bins - 1) / (I_max - I_min + 1e-8)
        scale_G = (n_bins - 1) / (G_max - G_min + 1e-8)

        c_lo_lo = dH_dH2d_raw[I_lo, G_lo];  c_hi_lo = dH_dH2d_raw[I_hi, G_lo]
        c_lo_hi = dH_dH2d_raw[I_lo, G_hi];  c_hi_hi = dH_dH2d_raw[I_hi, G_hi]

        dH_dI = W * scale_I * ((c_hi_lo*(1-wg) + c_hi_hi*wg) - (c_lo_lo*(1-wg) + c_lo_hi*wg))
        dH_dG = W * scale_G * ((c_lo_hi*(1-wi) + c_hi_hi*wi) - (c_lo_lo*(1-wi) + c_hi_lo*wi))

        grad_data_p = M_free.T @ (dH_dI * v_opt + dH_dG * g_opt)
        grad_data_q = grad_data_p * fp

        reg_loss, g_reg_q = self._reg_and_grad(
            q, free_idx, ns, nd, lam_smooth, lam_drift, alpha=1.0
        )
        return H_ent + reg_loss, (grad_data_q + g_reg_q).astype(np.float64)

    def _apply_field(self, field):
        """Scale and apply the correction field to the image."""
        corrected = self.pixel_array * field
        scale     = np.sum(self.pixel_array) / (np.sum(corrected) + 1e-10)
        corrected *= scale
        self._last_field = field
        return self.__class__(pixel_array=corrected)

    def correct_sd(self, ns=18, nd=13, field_bounds=(0.2, 3.0), **kwargs):
        """
        Fast profile correction.

        Estimates target intensity from deep tissue and scales pixels accordingly.

        Parameters
        ----------
        ns, nd : int
            Number of spline knots along s and d (default 18, 13).
        field_bounds : (float, float)
            Min/max clip range (default (0.2, 3.0)).

        Returns
        -------
        corrected_image : SDCorrectedImage
        field : (H, W) array
            The 2D correction field.
        """
        from scipy.interpolate import RectBivariateSpline

        self._build_sd_maps(**kwargs)
        mask   = self._sd_mask
        s_norm = self._sd_s_norm
        d_norm = self._sd_d_norm

        sc, dc = self._compute_adaptive_knots(ns, nd, **kwargs)

        bin_grid = self._build_heuristic_grid(sc, dc, ns, nd,
                                                field_bounds=field_bounds,
                                                **kwargs)

        spline = RectBivariateSpline(sc, dc, bin_grid, kx=3, ky=3)

        s_all = s_norm[mask].astype(np.float64)
        d_all = d_norm[mask].astype(np.float64)

        field = np.ones(self.pixel_array.shape, dtype=np.float64)
        field[mask] = np.clip(spline.ev(s_all, d_all), field_bounds[0], field_bounds[1])

        return self._apply_field(field), field

    def correct_sd_optimized(
        self,
        objective          = 'std',
        ns                 = 18,
        nd                 = 13,
        opt_stride         = 4,
        lam_smooth         = 0.15,
        lam_drift          = 0.10,
        field_bounds       = (0.2, 3.0),
        max_iter           = 300,
        ftol               = 1e-9,
        gtol               = 1e-6,
        depth_threshold    = 0.35,
        n_bins             = 32,
        hist_smooth_sigma  = 1.5,
        adaptive_bandwidth = True,
        _q0                = None,
        solver             = 'lbfgsb',
        bc_radius_px       = 0.0,
        d_bc_thresh        = 0.90,
        **kwargs
    ):
        """
        Fits the (s, d) correction field using L-BFGS-B or Trust-Region optimization.

        Parameters
        ----------
        solver : str
            'lbfgsb' or 'trust-ncg' (latter objective='std' only).
        bc_radius_px : float
            Radius for merging deep knots at the medial axis (default 0.0).
        d_bc_thresh : float
            Normalised depth for applying boundary conditions (default 0.90).
        hist_smooth_sigma : float
            Gaussian smoothing sigma in histogram bins (default 1.5).
        adaptive_bandwidth : bool
            Use Silverman's rule to adjust bandwidth (default True).
        """
        # Rest of docstring parameters are implicit via **kwargs in many calls,
        # but the core ones are handled by the caller or defaults.
        from scipy.optimize import minimize
        from scipy.interpolate import RectBivariateSpline
        from scipy.ndimage import sobel
        import time

        if objective not in ('std', 'entropy'):
            raise ValueError(f"objective must be 'std' or 'entropy', got '{objective}'.")
        if solver not in ('lbfgsb', 'trust-ncg'):
            raise ValueError(f"solver must be 'lbfgsb' or 'trust-ncg', got '{solver}'.")
        if solver == 'trust-ncg' and objective != 'std':
            raise ValueError("solver='trust-ncg' is only supported with objective='std'.")

        print("Building the s-d-map")
        t_build = time.time()
        self._build_sd_maps(**kwargs)
        print(f"  done in {time.time()-t_build:.1f}s", flush=True)

        mask   = self._sd_mask
        s_norm = self._sd_s_norm
        d_norm = self._sd_d_norm
        rows, cols = self.pixel_array.shape

        kx = min(3, ns - 1)
        ky = min(3, nd - 1)

        print("Computing knot locations")
        t_knots = time.time()
        sc, dc = self._compute_adaptive_knots(ns, nd, **kwargs)
        print(f"  done in {time.time()-t_knots:.1f}s", flush=True)

        opt_r, opt_c = np.where(mask[::opt_stride, ::opt_stride])
        opt_r = np.clip(opt_r * opt_stride, 0, rows - 1)
        opt_c = np.clip(opt_c * opt_stride, 0, cols - 1)

        s_opt = s_norm[opt_r, opt_c].astype(np.float64)
        d_opt = d_norm[opt_r, opt_c].astype(np.float64)
        v_opt = self.pixel_array[opt_r, opt_c].astype(np.float64)
        n_par = ns * nd

        print(f"Building M ({len(v_opt)} pixels × {n_par} params)…", flush=True)
        t0 = time.time()
        M_free, free_idx, fixed_contrib, bc_idx = self._build_interp_matrix(
            sc, dc, s_opt, d_opt, ns, nd
        )
        print(f"  done in {time.time()-t0:.1f}s  ({len(free_idx)} free params)", flush=True)

        self._last_sc       = sc
        self._last_dc       = dc

        # Medial-axis boundary conditions
        merge_map = {}
        if bc_radius_px > 0.0:
            knot_xy = self._compute_knot_positions(
                sc, dc, opt_r, opt_c, s_opt, d_opt
            )
            self._last_knot_xy = knot_xy
            M_free, free_idx, merge_map, bc_keep = self._apply_medial_boundary_conditions(
                knot_xy, dc, M_free, free_idx,
                n_par, ns, nd,
                d_bc_thresh=d_bc_thresh,
                bc_radius_px=bc_radius_px,
            )
            # If q0 was provided externally (e.g. from hierarchical upsampling),
            # it was built against the pre-BC free_idx and must be trimmed to
            # match the reduced post-BC free_idx, otherwise bounds length != len(q0).
            if _q0 is not None:
                _q0 = _q0[bc_keep]
        self._last_merge_map = merge_map
        # Store post-BC free_idx so the hierarchical loop reads the correct
        # length when building q0 for the next level via _upsample_grid.
        self._last_free_idx = free_idx

        if _q0 is not None:
            q0 = _q0
        else:
            print("Computing the heuristic grid")
            t_grid = time.time()
            bin_grid = self._build_heuristic_grid(sc, dc, ns, nd,
                                                field_bounds=field_bounds, **kwargs)
            print(f"  done in {time.time()-t_grid:.1f}s", flush=True)
            p0 = np.clip(bin_grid.ravel()[free_idx].copy(), 1e-6, None)
            q0 = np.log(p0)

        if objective == 'std':
            def obj_fn(q):
                return self._objective_std(q, M_free, fixed_contrib, v_opt,
                                        free_idx, ns, nd, lam_smooth, lam_drift)
        else:
            gr_full = np.sqrt(sobel(self.pixel_array, axis=0)**2 +
                            sobel(self.pixel_array, axis=1)**2)
            g_opt        = gr_full[opt_r, opt_c].astype(np.float64)
            d_opt_n      = d_norm[opt_r, opt_c]
            depth_weight = np.clip(
                1.0 - (d_opt_n - depth_threshold) / (1.0 - depth_threshold), 0.05, 1.0
            )
            def obj_fn(q):
                return self._objective_entropy(q, M_free, fixed_contrib, v_opt, g_opt,
                                            free_idx, ns, nd, lam_smooth, lam_drift,
                                            depth_weight, n_bins, hist_smooth_sigma,
                                            adaptive_bandwidth)

        print(f"Optimizing [{objective}] with [{solver}]…", flush=True)
        t1 = time.time()

        if solver == 'trust-ncg':
            def hessp_fn(q, vec):
                return self._hessp_std(q, vec, M_free, fixed_contrib, v_opt,
                                    free_idx, ns, nd, lam_smooth, lam_drift)
            result = minimize(
                obj_fn, q0, jac=True, method='trust-ncg',
                hessp=hessp_fn,
                options={'maxiter': max_iter, 'gtol': gtol}
            )
        else:
            log_lb = np.log(field_bounds[0]) if field_bounds[0] > 0 else -np.inf
            log_ub = np.log(field_bounds[1]) if np.isfinite(field_bounds[1]) else np.inf
            log_bounds = [(log_lb, log_ub)] * len(free_idx)
            result = minimize(
                obj_fn, q0, jac=True, method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )

        print(f"  {result.message}  iters={result.nit}  time={time.time()-t1:.1f}s",
            flush=True)

        opt_par            = np.ones(n_par, dtype=np.float64)
        opt_par[free_idx]  = np.exp(result.x)
        # Restore merged (follower) knots from their leader's optimised value
        for follower_flat, leader_flat in merge_map.items():
            opt_par[follower_flat] = opt_par[leader_flat]

        spline_opt = RectBivariateSpline(sc, dc, opt_par.reshape(ns, nd), kx=kx, ky=ky)
        s_all = s_norm[mask].astype(np.float64)
        d_all = d_norm[mask].astype(np.float64)

        field = np.ones((rows, cols), dtype=np.float64)
        field[mask] = np.clip(spline_opt.ev(s_all, d_all),
                            field_bounds[0], field_bounds[1])

        return self._apply_field(field), field, result

    def _upsample_grid(self, p_opt_coarse, free_idx_coarse,
                    sc_coarse, dc_coarse,
                    sc_fine,   dc_fine,
                    ns_coarse, nd_coarse,
                    ns_fine,   nd_fine,
                    free_idx_fine):
        """
        Upsamples a coarse grid solution to initialize a fine grid.

        Parameters
        ----------
        p_opt_coarse  : (n_free_coarse,) optimal p values from coarse level
        free_idx_coarse / _fine : index arrays of free (non-BC) knots
        sc_coarse, dc_coarse    : coarse knot centres (1-D arrays)
        sc_fine,   dc_fine      : fine   knot centres (1-D arrays)
        ns_*, nd_*              : grid dimensions at each level

        Returns
        -------
        q0_fine : (n_free_fine,) warm-start in log-space for the fine level
        """
        from scipy.interpolate import RectBivariateSpline

        # Reconstruct the full coarse grid.
        # free_idx_coarse is the post-BC index array: it contains only leader
        # flat indices (followers were removed by _apply_medial_boundary_conditions).
        # Followers must be restored from their leader before we can reshape into
        # a 2-D grid for spline fitting.
        n_par_coarse  = ns_coarse * nd_coarse
        p_full_coarse = np.ones(n_par_coarse, dtype=np.float64)
        p_full_coarse[free_idx_coarse] = p_opt_coarse

        # Propagate leaders → followers using the stored merge_map
        merge_map = getattr(self, '_last_merge_map', {}) or {}
        for follower_flat, leader_flat in merge_map.items():
            if follower_flat < n_par_coarse:
                p_full_coarse[follower_flat] = p_full_coarse[leader_flat]

        G_coarse = p_full_coarse.reshape(ns_coarse, nd_coarse)

        # Fit a spline to the coarse grid values
        spl = RectBivariateSpline(sc_coarse, dc_coarse, G_coarse, kx=3, ky=3)

        # Evaluate at all fine knot positions
        n_par_fine = ns_fine * nd_fine
        p_full_fine = np.ones(n_par_fine, dtype=np.float64)

        ss, dd = np.meshgrid(sc_fine, dc_fine, indexing='ij')   # (ns_fine, nd_fine)
        G_fine = spl.ev(ss.ravel(), dd.ravel()).reshape(ns_fine, nd_fine)

        # Clip to a safe positive range before taking log
        G_fine = np.clip(G_fine, 1e-6, None)
        p_full_fine = G_fine.ravel()

        # Return only the free (non-BC) knots, encoded in log-space
        q0_fine = np.log(p_full_fine[free_idx_fine])
        return q0_fine
    
    def correct_sd_optimized_hierarchical(
        self,
        objective         = 'entropy',
        levels            = None,
        opt_stride        = 4,
        lam_smooth        = 0.15,
        lam_drift         = 0.10,
        field_bounds      = (0.2, 3.0),
        max_iter_coarse   = 100,
        max_iter_fine     = 300,
        ftol              = 1e-9,
        gtol              = 1e-6,
        depth_threshold   = 0.35,
        n_bins            = 32,
        hist_smooth_sigma = 1.5,
        **kwargs
    ):
        """
        Hierarchical multi-resolution correction.

        Starts on a coarse grid and refines it, reducing the risk of local minima.

        Parameters
        ----------
        levels : list of (ns, nd) tuples
            Grid resolution stages.
        max_iter_coarse, max_iter_fine : int
            Iteration budgets.

        Returns
        -------
        corrected_image : SDCorrectedImage
        field : (H, W) array
        results : list of OptimizeResult, one per level
        """
        if levels is None:
            levels = [(6, 5), (13, 8)]

        # Build sd-maps once here.  Every subsequent call to _build_sd_maps
        # inside correct_sd_optimized will find the cache valid and return
        # immediately (no recomputation).
        self._build_sd_maps(**kwargs)

        shared = dict(
            objective         = objective,
            opt_stride        = opt_stride,
            lam_smooth        = lam_smooth,
            lam_drift         = lam_drift,
            field_bounds      = field_bounds,
            ftol              = ftol,
            gtol              = gtol,
            depth_threshold   = depth_threshold,
            n_bins            = n_bins,
            hist_smooth_sigma = hist_smooth_sigma,
        )

        q_prev        = None   # log-space solution from the previous level
        free_idx_prev = None
        sc_prev       = None
        dc_prev       = None
        ns_prev       = None
        nd_prev       = None
        results       = []

        for level_idx, (ns, nd) in enumerate(levels):
            is_last  = (level_idx == len(levels) - 1)
            max_iter = max_iter_fine if is_last else max_iter_coarse

            print(f"\nLevel {level_idx + 1}/{len(levels)}  "
                f"grid={ns}×{nd}  "
                f"({'final' if is_last else 'coarse'})", flush=True)

            # Warm start
            if q_prev is None:
                # First level: correct_sd_optimized will use its heuristic default.
                q0 = None
            else:
                # Compute this level's knot positions (cheap — just quantile maths,
                # no matrix build).  correct_sd_optimized will compute them again
                # internally, but the duplication costs only milliseconds.
                sc_cur, dc_cur = self._compute_adaptive_knots(ns, nd, **kwargs)

                # free_idx for the current level: since _build_interp_matrix
                # currently pins no BCs, this is always np.arange(ns * nd).
                # We read it from the cache written by the previous
                # correct_sd_optimized call, which is the robust way to get it
                # should BCs ever be re-activated in _build_interp_matrix.
                # For the very next level we must compute it before the call, so
                # we derive it directly from ns/nd here.
                free_idx_cur = np.arange(ns * nd)

                q0 = self._upsample_grid(
                    np.exp(q_prev), free_idx_prev,
                    sc_prev, dc_prev,
                    sc_cur,  dc_cur,
                    ns_prev, nd_prev,
                    ns,      nd,
                    free_idx_cur,
                )

            # Run the existing single-level optimiser
            # M is built exactly once here; _build_sd_maps hits the cache.
            # After this call, self._last_sc / _dc / _free_idx hold this level's
            # knots and free indices, ready for the next upsample step.
            _, field, result = self.correct_sd_optimized(
                ns       = ns,
                nd       = nd,
                max_iter = max_iter,
                _q0      = q0,
                **shared,
                **kwargs,
            )
            results.append(result)

            # Carry solution forward — read from cache, no rebuild
            q_prev        = result.x            # already in log-space
            sc_prev       = self._last_sc
            dc_prev       = self._last_dc
            free_idx_prev = self._last_free_idx
            ns_prev, nd_prev = ns, nd

        corrected_image = self._apply_field(field)
        return corrected_image, field, results

    def visualize_correction(
        self,
        field,
        corrected_image,
        ns              = 18,
        nd              = 13,
        sc              = None,
        dc              = None,
        cmap_field      = 'RdBu_r',
        field_vmin      = None,
        field_vmax      = None,
        figsize         = (22, 14),
        save_path       = None,
        dpi             = 150,
        merge_map       = None,
        **kwargs
    ):
        """
        Plots the correction field, the spline grid, and intensity profiles.

        Parameters
        ----------
        field : (H, W) array
            The correction field used.
        corrected_image : SDCorrectedImage
            The result of applying the field.
        ns, nd : int
            Spline grid dimensions.
        sc, dc : arrays, optional
            Explicit knot centers.
        save_path : str, optional
            Output image path.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
        from collections import defaultdict
        from scipy.interpolate import RectBivariateSpline
        from scipy.spatial import cKDTree

        # Resolve merge_map: explicit arg → instance attribute → empty dict
        if merge_map is None:
            merge_map = getattr(self, '_last_merge_map', None) or {}

        self._build_sd_maps(**kwargs)
        mask   = self._sd_mask
        s_norm = self._sd_s_norm
        d_norm = self._sd_d_norm

        # Knot centers and bin edges
        if sc is None or dc is None:
            sc, dc = self._compute_adaptive_knots(ns, nd, **kwargs)

        se = np.concatenate([[0.0], 0.5 * (sc[:-1] + sc[1:]), [1.0]])
        de = np.concatenate([[0.0], 0.5 * (dc[:-1] + dc[1:]), [1.0]])

        # bin field onto grid values
        s_all = s_norm[mask].astype(np.float64)
        d_all = d_norm[mask].astype(np.float64)
        f_all = field[mask].astype(np.float64)

        grid_values = np.ones((ns, nd), dtype=np.float64)
        for i in range(ns):
            for j in range(nd):
                in_cell = (
                    (s_all >= se[i]) & (s_all < se[i + 1]) &
                    (d_all >= de[j]) & (d_all < de[j + 1])
                )
                if in_cell.any():
                    grid_values[i, j] = f_all[in_cell].mean()

        # Fine-grid spline for display
        spline_display = RectBivariateSpline(sc, dc, grid_values, kx=3, ky=3)
        n_fine   = 200
        s_fine   = np.linspace(sc[0], sc[-1], n_fine)
        d_fine   = np.linspace(dc[0], dc[-1], n_fine)
        field_sd = spline_display(s_fine, d_fine)

        # Map knots to image space using a strip-based centroid approach
        knot_r = np.zeros((ns, nd), dtype=int)
        knot_c = np.zeros((ns, nd), dtype=int)
        all_r, all_c = np.where(mask)
        s_px = s_norm[all_r, all_c]
        d_px = d_norm[all_r, all_c]
        k_near = 20   # number of closest-in-d pixels to average per knot
        for i in range(ns):
            in_strip = (s_px >= se[i]) & (s_px < se[i + 1])
            if not in_strip.any():
                # Widen to full mask as last resort (should be very rare)
                in_strip = np.ones(len(s_px), dtype=bool)
            strip_r = all_r[in_strip]
            strip_c = all_c[in_strip]
            strip_d = d_px[in_strip]
            for j in range(nd):
                k = min(k_near, len(strip_r))
                order = np.argsort(np.abs(strip_d - dc[j]))[:k]
                knot_r[i, j] = int(np.round(strip_r[order].mean()))
                knot_c[i, j] = int(np.round(strip_c[order].mean()))

        # Color limits
        f_masked = field[mask]
        dev = max(abs(float(f_masked.max()) - 1.0),
                abs(float(f_masked.min()) - 1.0))
        if field_vmin is None:
            field_vmin = 1.0 - dev
        if field_vmax is None:
            field_vmax = 1.0 + dev
        norm_f = mcolors.Normalize(vmin=field_vmin, vmax=field_vmax)
        cmap_f = plt.get_cmap(cmap_field)

        img_orig = self.pixel_array.astype(np.float64)
        img_corr = corrected_image.pixel_array.astype(np.float64)
        vmin_img, vmax_img = np.nanpercentile(img_orig[mask], [1, 99])

        field_display = np.where(mask, field, np.nan)
        kv_flat = grid_values.ravel()
        kr_flat = knot_r.ravel()
        kc_flat = knot_c.ravel()

        # Figure
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        ax_orig    = fig.add_subplot(gs[0, 0])
        ax_corr    = fig.add_subplot(gs[0, 1])
        ax_field   = fig.add_subplot(gs[0, 2])
        ax_knots   = fig.add_subplot(gs[1, 0])
        ax_sd      = fig.add_subplot(gs[1, 1])
        ax_profile = fig.add_subplot(gs[1, 2])

        # [0,0] Original image
        ax_orig.imshow(img_orig, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                    origin='upper', interpolation='nearest')
        ax_orig.set_title('Original image', fontsize=11)
        ax_orig.axis('off')

        # [0,1] Corrected image
        ax_corr.imshow(img_corr, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                    origin='upper', interpolation='nearest')
        ax_corr.set_title('Corrected image', fontsize=11)
        ax_corr.axis('off')

        # [0,2] Field on breast + knot scatter
        ax_field.imshow(img_orig, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                        origin='upper', interpolation='nearest', alpha=0.45)
        im_f = ax_field.imshow(field_display, cmap=cmap_f, norm=norm_f,
                            origin='upper', interpolation='nearest', alpha=0.75)
        ax_field.scatter(kc_flat, kr_flat, c=kv_flat, cmap=cmap_f, norm=norm_f,
                        s=30, edgecolors='k', linewidths=0.5, zorder=5)
        cb_f = fig.colorbar(im_f, ax=ax_field, fraction=0.046, pad=0.04)
        cb_f.set_label('f(x,y)', fontsize=9)
        cb_f.locator = MaxNLocator(nbins=5); cb_f.update_ticks()
        ax_field.set_title(f'Field on breast ({ns}x{nd} knots)', fontsize=11)
        ax_field.axis('off')

        # [1,0] Knot grid in image space — edge colour encodes BC group membership
        ax_knots.imshow(img_orig, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                        origin='upper', interpolation='nearest', alpha=0.65)
        for i in range(ns):
            ax_knots.plot(knot_c[i, :], knot_r[i, :],
                        color='steelblue', linewidth=0.7, alpha=0.6, zorder=3)
        for j in range(nd):
            ax_knots.plot(knot_c[:, j], knot_r[:, j],
                        color='darkorange', linewidth=0.7, alpha=0.6, zorder=3)

        # Build per-knot edge colour from merge_map
        # groups: leader_flat → set of all member flat indices (including leader)
        groups = defaultdict(set)
        for follower, leader in merge_map.items():
            groups[leader].add(leader)
            groups[leader].add(follower)
        group_list   = sorted(groups.keys())
        n_groups     = len(group_list)
        q_cmap       = plt.get_cmap('tab10' if n_groups <= 10 else 'tab20')
        group_colors = [q_cmap(k / max(n_groups, 1)) for k in range(n_groups)]
        flat_to_group = {}
        for g_idx, leader in enumerate(group_list):
            for member in groups[leader]:
                flat_to_group[member] = g_idx

        n_knots   = ns * nd
        # Draw knots one-by-one so each can carry its own edge lw
        for k in range(n_knots):
            if k in flat_to_group:
                g_idx = flat_to_group[k]
                ec    = group_colors[g_idx]
                lw    = 2.0
                sz    = 55
            else:
                ec    = 'black'
                lw    = 0.5
                sz    = 28
            ax_knots.scatter(
                kc_flat[k], kr_flat[k],
                c=[cmap_f(norm_f(kv_flat[k]))],
                s=sz, edgecolors=[ec], linewidths=lw, zorder=5,
            )

        # Legend: grid lines + one patch per BC group
        legend_handles = [
            Line2D([0], [0], color='steelblue',  lw=1.2, label=f'iso-s ({ns})'),
            Line2D([0], [0], color='darkorange', lw=1.2, label=f'iso-d ({nd})'),
        ]
        # The merge_map flat indices were encoded as  flat = i * nd_opt + j
        # where nd_opt is the nd used during optimization (self._last_dc), which
        # may differ from the visualization's nd parameter (e.g. in hierarchical
        # runs where the final level has a different grid than what is plotted).
        nd_opt = len(self._last_dc) if hasattr(self, '_last_dc') and self._last_dc is not None else nd
        for g_idx, leader in enumerate(group_list):
            members  = sorted(groups[leader])
            labels = ', '.join(f's{m // nd_opt}d{m % nd_opt}' for m in members)
            legend_handles.append(mpatches.Patch(
                facecolor='none',
                edgecolor=group_colors[g_idx],
                linewidth=2.0,
                label=f'BC group {g_idx + 1}: [{labels}]',
            ))

        ax_knots.legend(handles=legend_handles,
                        loc='upper right', fontsize=7.0, framealpha=0.80)
        title_bc = f' | {n_groups} BC group(s)' if n_groups else ''
        ax_knots.set_title(f'Knot grid in image space{title_bc}', fontsize=11)
        ax_knots.axis('off')

        # [1,1] Field in (s, d) space
        im_sd = ax_sd.imshow(
            field_sd.T, origin='lower',
            extent=[s_fine[0], s_fine[-1], d_fine[0], d_fine[-1]],
            aspect='auto', cmap=cmap_f, norm=norm_f, interpolation='bilinear',
        )
        for i in range(ns):
            ax_sd.axvline(sc[i], color='steelblue',  linewidth=0.7, alpha=0.5)
        for j in range(nd):
            ax_sd.axhline(dc[j], color='darkorange', linewidth=0.7, alpha=0.5)
        # Knot scatter in (s,d) space
        ss_grid, dd_grid = np.meshgrid(sc, dc, indexing='ij')
        ax_sd.scatter(ss_grid.ravel(), dd_grid.ravel(),
                    c=grid_values.ravel(), cmap=cmap_f, norm=norm_f,
                    s=30, edgecolors='k', linewidths=0.5, zorder=5)
        cb_sd = fig.colorbar(im_sd, ax=ax_sd, fraction=0.046, pad=0.04)
        cb_sd.set_label('f(s, d)', fontsize=9)
        cb_sd.locator = MaxNLocator(nbins=5); cb_sd.update_ticks()
        ax_sd.set_xlabel('s  (normalised arc-length)', fontsize=9)
        ax_sd.set_ylabel('d  (normalised depth)', fontsize=9)
        ax_sd.set_title('Field in unfolded (s, d) space', fontsize=11)
        ax_sd.tick_params(labelsize=8)

        # [1,2] Mean field profile vs depth
        n_bins_prof = 40
        d_edges   = np.linspace(0, 1, n_bins_prof + 1)
        d_centres = 0.5 * (d_edges[:-1] + d_edges[1:])
        f_mean    = np.full(n_bins_prof, np.nan)
        f_std     = np.full(n_bins_prof, np.nan)
        d_flat    = d_norm[mask]
        for b in range(n_bins_prof):
            in_bin = (d_flat >= d_edges[b]) & (d_flat < d_edges[b + 1])
            if in_bin.sum() > 5:
                vals       = f_masked[in_bin]
                f_mean[b]  = vals.mean()
                f_std[b]   = vals.std()

        ray_colors = plt.get_cmap('tab20')(np.linspace(0, 1, ns))

        ax_profile.axhline(1.0, color='gray', linewidth=0.8,
                        linestyle='--', label='No correction')
        ax_profile.fill_between(
            d_centres,
            np.where(np.isnan(f_mean), np.nan, f_mean - f_std),
            np.where(np.isnan(f_mean), np.nan, f_mean + f_std),
            alpha=0.25, color='steelblue', label='±1 std',
        )

        # Per-ray profiles: pixels in each s-strip, binned along d
        for i in range(ns):
            in_strip = (s_all >= se[i]) & (s_all < se[i + 1])
            ray_mean = np.full(n_bins_prof, np.nan)
            for b in range(n_bins_prof):
                sel = in_strip & (d_all >= d_edges[b]) & (d_all < d_edges[b + 1])
                if sel.sum() > 3:
                    ray_mean[b] = f_all[sel].mean()
            lbl = f's={sc[i]:.2f}' if ns <= 10 else ('s-rays' if i == 0 else None)
            ax_profile.plot(
                d_centres, ray_mean,
                color=ray_colors[i], linewidth=0.9, alpha=0.65, label=lbl,
            )

        ax_profile.plot(d_centres, f_mean, color='steelblue',
                        linewidth=2.0, label='Mean f', zorder=5)
        ax_profile.set_xlabel('d (normalized depth)', fontsize=9)
        ax_profile.set_ylabel('Correction factor f', fontsize=9)
        ax_profile.set_xlim(0, 1)
        ax_profile.set_ylim(field_vmin - 0.05, field_vmax + 0.05)
        ax_profile.legend(fontsize=7, loc='upper right',
                          title=(f'{ns} s-rays' if ns > 10 else None))
        ax_profile.set_title('Mean field profile vs depth', fontsize=11)
        ax_profile.tick_params(labelsize=8)
        ax_profile.grid(True, linewidth=0.4, alpha=0.5)

        fig.suptitle(
            f'(s, d) Correction Result — spline grid {ns}x{nd}'
            f' | field range [{f_masked.min():.3f}, {f_masked.max():.3f}]',
            fontsize=13, fontweight='bold', y=1.01,
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f'Saved to {save_path}')
        plt.show()
        return fig, (ax_orig, ax_corr, ax_field, ax_knots, ax_sd, ax_profile)
    

if __name__=="__main__":
    img = SDCorrectedImage(dicom_path=r"C:\Users\Santeri\OneDrive - University of Helsinki\Desktop\BIBCorrection\Non-suppressed\85\I1225")
    img._compute_adaptive_knots(15,10)