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
        from scipy.ndimage import distance_transform_edt, binary_erosion, gaussian_filter, gaussian_filter1d
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

        # Smooth the contour to remove pixel-grid kinks that cause s-jumps
        boundary_smooth_sigma = kwargs.get('boundary_smooth_sigma', 250.0)
        if boundary_smooth_sigma > 0:
            boundary = np.stack([
                gaussian_filter1d(boundary[:, 0], sigma=boundary_smooth_sigma),
                gaussian_filter1d(boundary[:, 1], sigma=boundary_smooth_sigma),
            ], axis=1)

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
        br = np.clip(boundary.astype(int)[:, 0], 0, mask.shape[0] - 1)
        bc = np.clip(boundary.astype(int)[:, 1], 0, mask.shape[1] - 1)
        boundary_mask[br,bc] = True
        # return boundary_mask

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
        # Normalized arc-length
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
                d_margin=0.01,
                d_strategy='gradient',
                d_beta_a=0.6,
                d_beta_b=2.5,
                knot_smooth_sigma=5.0,
                knot_weight_floor=0.05,
                top_bottom_cut=.25,
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
            
            height,_ = mask.shape
            h0 = int(top_bottom_cut*height)
            h1 = int((1-top_bottom_cut)*height)
            cut_mask = mask[h0:h1,:]
            
            d_flat = self._sd_d_norm[h0:h1,:]
            d_flat = d_flat[cut_mask].astype(np.float64)


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

            # d-knots
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

                # if hasattr(self, '_last_field') and self._last_field is not None:
                #     f_flat = self._last_field[mask].astype(np.float64)
                # else:
                    # v      = self.pixel_array[mask].astype(np.float64)
                v = self.pixel_array[h0:h1,:]
                v      = v[cut_mask].astype(np.float64)
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

                cdf = np.cumsum(weight)
                cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
                dc = np.interp(np.linspace(0.0, 1.0, nd), cdf, d_grid)

            elif d_strategy == 'beta':
                uniform = np.linspace(0.0, 1.0, nd)
                warped  = beta_dist.ppf(uniform, d_beta_a, d_beta_b)
                dc      = d_lo + warped * (d_hi - d_lo)

            else:
                raise ValueError(
                    f"d_strategy must be 'quantile', 'gradient', 'curvature', or "
                    f"'beta'; got '{d_strategy}'."
                )

            # Enforce strict monotonicity
            dc[0]  = d_lo;  dc[-1]  = d_hi
            dc = np.maximum.accumulate(dc)
            dc = np.minimum.accumulate(dc[::-1])[::-1]

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

    def _apply_medial_boundary_conditions(
            self, knot_xy, dc, M_free, free_idx,
            n_pars, ns, nd, d_bc_thresh, bc_radius_px):
        """
        Forces coincident deep knots to share the same value.

        Builds a weighted graph over all free knots whose depth coordinate
        exceeds d_bc_thresh.  Each node is a knot, each edge carries the
        Euclidean pixel-space distance between the two endpoints.  Connected
        components are found by running BFS on the subgraph that contains
        only edges with weight <= bc_radius_px.  Every knot in a component
        is merged into a single representative.

        Parameters
        ----------
        knot_xy     : (ns, nd, 2) array — pixel-space (row, col) of every knot
        dc          : (nd,) array    — normalised depth of each depth column
        M_free      : (n_pix, n_free) — B-spline interpolation matrix
        free_idx    : (n_free,) int  — flat index i*nd+j for each column of M_free
        n_par       : int            — ns * nd
        ns, nd      : int            — grid dimensions
        d_bc_thresh : float          — minimum depth to consider for merging
        bc_radius_px: float          — merge threshold in pixels

        Returns
        -------
        M_free_bc   : (n_pix, n_free_bc) — reduced interpolation matrix
        free_idx_bc : (n_free_bc,) int   — flat indices of the surviving knots
        merge_map   : dict {follower_flat: leader_flat}
        keep        : (n_free,) bool     — True for surviving columns
        """
        import numpy as np
        from collections import deque

        M_work    = M_free.copy()
        flat_to_k = {int(f): k for k, f in enumerate(free_idx)}
        merge_map = {}
        remove_k  = set()

        nodes = []
        for j in range(nd):
            if float(dc[j]) < d_bc_thresh:
                continue
            for i in range(ns):
                flat = int(i * nd + j)
                if flat in flat_to_k:
                    nodes.append((flat, flat_to_k[flat], knot_xy[i, j]))

        if len(nodes) < 2:
            keep = np.ones(len(free_idx), dtype=bool)
            print(f"  Medial BC: fewer than 2 candidate knots at "
                f"d >= {d_bc_thresh:.2f} — nothing to merge.", flush=True)
            return M_free, free_idx, merge_map, keep

        n_nodes   = len(nodes)
        positions = np.array([pos for _, _, pos in nodes])   # (n_nodes, 2)

        # Build the weighted distance matrix
        delta      = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(delta, axis=-1)          # (n_nodes, n_nodes)

        # Build adjacency list
        adjacency = [[] for _ in range(n_nodes)]
        for a in range(n_nodes):
            for b in range(a + 1, n_nodes):
                if dist_matrix[a, b] <= bc_radius_px:
                    adjacency[a].append(b)
                    adjacency[b].append(a)

        # BFS to find connected components
        visited    = [False] * n_nodes
        components = []

        for start in range(n_nodes):
            if visited[start]:
                continue
            component = []
            queue = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbour in adjacency[node]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        queue.append(neighbour)
            components.append(component)

        # Merge each multi-node component
        for component in components:
            if len(component) < 2:
                continue

            flat_a, k_a, _ = nodes[component[0]]          # leader

            for local_idx in component[1:]:                # followers
                flat_b, k_b, _ = nodes[local_idx]
                M_work[:, k_a] += M_work[:, k_b]          # absorb follower column
                merge_map[flat_b] = flat_a
                remove_k.add(k_b)

        # Build the reduced matrix and index array
        keep = np.ones(len(free_idx), dtype=bool)
        for k in remove_k:
            keep[k] = False

        M_free_bc   = M_work[:, keep]
        free_idx_bc = free_idx[keep]

        n_merged = int(keep.size - keep.sum())
        n_groups  = sum(1 for c in components if len(c) > 1)
        if n_merged:
            print(f"  Medial BC: merged {n_merged} follower knot(s) across "
                f"{n_groups} component(s) → "
                f"{len(free_idx_bc)} free params (was {len(free_idx)})", flush=True)
        else:
            print(f"  Medial BC: no knots within {bc_radius_px:.0f}px at "
                f"d >= {d_bc_thresh:.2f} — try increasing bc_radius_px.", flush=True)

        return M_free_bc, free_idx_bc, merge_map, keep
    
    def _apply_seam_boundary_conditions(
            self, knot_xy, dc, M_free, free_idx,
            n_par, ns, nd,
            seam_radius_px,
            seam_s_threshold=0.05):
        """
        Merges knots that are physically close to the s-coordinate seam.

        The s-map is built from an open skin contour whose two endpoints
        (s=0 and s=1) both sit at the chest-wall attachment.  Inside the
        breast the nearest-contour-point propagation creates a Voronoi seam:
        a connected band of pixels where s_norm jumps discontinuously from
        ~1 to ~0.  Any knot whose pixel-space position is within
        seam_radius_px of this seam band is a candidate for merging,
        regardless of which s-strip (i index) it belongs to.

        The algorithm
        -------------
        1. Identify seam pixels: masked pixels where s_norm < seam_s_threshold
        OR s_norm > 1 - seam_s_threshold.  These pixels lie in the thin
        Voronoi zone adjacent to the discontinuity.
        2. Build a KDTree over those pixels for O(log n) distance queries.
        3. For every free knot, query its pixel-space distance to the nearest
        seam pixel.
        4. Group candidate knots (distance <= seam_radius_px) by depth level j.
        5. Within each depth level run BFS on the pixel-distance proximity
        graph (edge iff distance <= seam_radius_px) to find connected
        components, then merge each component into a single leader — exactly
        the same column-absorption used by the medial BC.

        Parameters
        ----------
        knot_xy          : (ns, nd, 2) array — pixel (row, col) per knot
        dc               : (nd,) array       — normalised depth per column
        M_free           : (n_pix, n_free)   — B-spline interpolation matrix
        free_idx         : (n_free,) int     — flat index i*nd+j per column
        n_par            : int               — ns * nd
        ns, nd           : int               — grid dimensions
        seam_radius_px   : float             — merge threshold in pixels
        seam_s_threshold : float             — s_norm fraction that defines
                                            the seam zone (default 0.05,
                                            i.e. s < 0.05 or s > 0.95)

        Returns
        -------
        M_free_bc   : (n_pix, n_free_bc) — reduced interpolation matrix
        free_idx_bc : (n_free_bc,) int   — flat indices of surviving knots
        merge_map   : dict {follower_flat: leader_flat}
        keep        : (n_free,) bool     — True for surviving columns
        """
        import numpy as np
        from scipy.spatial import KDTree
        from collections import deque

        M_work    = M_free.copy()
        flat_to_k = {int(f): k for k, f in enumerate(free_idx)}
        merge_map = {}
        remove_k  = set()

        # ── 1. Build the seam pixel set ───────────────────────────────────────────
        #
        # Pixels with s_norm near 0 or 1 lie in the Voronoi zone adjacent to the
        # discontinuity.  We restrict to the breast mask so we don't pick up the
        # zero-padded background.

        mask   = self._sd_mask
        s_norm = self._sd_s_norm

        seam_pixel_mask = mask & (
            (s_norm < seam_s_threshold) | (s_norm > 1.0 - seam_s_threshold)
        )
        seam_pixels = np.argwhere(seam_pixel_mask)   # (M, 2) — (row, col)

        if len(seam_pixels) == 0:
            keep = np.ones(len(free_idx), dtype=bool)
            print(f"  Seam BC: no seam pixels found at threshold "
                f"{seam_s_threshold:.3f} — nothing to merge.", flush=True)
            return M_free, free_idx, merge_map, keep

        seam_tree = KDTree(seam_pixels)

        # ── 2. Find knots within seam_radius_px of the seam ──────────────────────
        #
        # For each free knot query the tree; candidates are grouped by depth j
        # so that only knots at the same depth level compete for merging.

        candidates_by_depth = {j: [] for j in range(nd)}

        for i in range(ns):
            for j in range(nd):
                flat = int(i * nd + j)
                if flat not in flat_to_k:
                    continue   # already removed by a prior BC pass

                pos  = knot_xy[i, j]                      # (row, col)
                dist_to_seam, _ = seam_tree.query(pos)

                if dist_to_seam <= seam_radius_px:
                    candidates_by_depth[j].append(
                        (flat, flat_to_k[flat], pos)
                    )

        # ── 3. BFS within each depth level to find connected components ───────────
        #
        # Two candidate knots are connected if their mutual pixel distance is also
        # <= seam_radius_px.  Every component is merged into its first member
        # (the leader).

        total_merged = 0
        merged_depths = []

        for j, nodes in candidates_by_depth.items():
            if len(nodes) < 2:
                continue

            n_nodes   = len(nodes)
            positions = np.array([pos for _, _, pos in nodes])

            # Pairwise distances among the candidates at this depth level
            delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(delta, axis=-1)          # (n_nodes, n_nodes)

            adjacency = [[] for _ in range(n_nodes)]
            for a in range(n_nodes):
                for b in range(a + 1, n_nodes):
                    if dists[a, b] <= seam_radius_px:
                        adjacency[a].append(b)
                        adjacency[b].append(a)

            visited    = [False] * n_nodes
            components = []
            for start in range(n_nodes):
                if visited[start]:
                    continue
                component = []
                queue = deque([start])
                visited[start] = True
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for nb in adjacency[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
                components.append(component)

            n_merged_j = 0
            for component in components:
                if len(component) < 2:
                    continue
                flat_a, k_a, _ = nodes[component[0]]       # leader
                for local_idx in component[1:]:             # followers
                    flat_b, k_b, _ = nodes[local_idx]
                    M_work[:, k_a] += M_work[:, k_b]
                    merge_map[flat_b] = flat_a
                    remove_k.add(k_b)
                    n_merged_j += 1

            if n_merged_j:
                merged_depths.append((j, float(dc[j]), n_merged_j))
                total_merged += n_merged_j

        # ── 4. Build the reduced matrix and index array ───────────────────────────

        keep = np.ones(len(free_idx), dtype=bool)
        for k in remove_k:
            keep[k] = False

        M_free_bc   = M_work[:, keep]
        free_idx_bc = free_idx[keep]

        if total_merged:
            depth_str = ', '.join(
                f'd={dc:.2f}({n}merged)' for _, dc, n in merged_depths
            )
            print(f"  Seam BC: merged {total_merged} knot(s) across "
                f"{len(merged_depths)} depth level(s): {depth_str} "
                f"→ {len(free_idx_bc)} free params (was {len(free_idx)})",
                flush=True)
        else:
            print(f"  Seam BC: no knots within {seam_radius_px:.0f}px of the "
                f"seam — try increasing seam_radius_px or seam_s_threshold.",
                flush=True)

        return M_free_bc, free_idx_bc, merge_map, keep

    def _reg_and_grad(self, q, free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift, debug=False):
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
        # ds_ = np.diff(q2d, axis=0)
        # dd_ = np.diff(q2d, axis=1)
        # smooth_loss = lam_smooth * (np.sum(ds_**2) + np.sum(dd_**2)) / n_par
        
        # dq = np.zeros((ns, nd), dtype=np.float64)
        # dq[1:,  :] += 2 * ds_;  dq[:-1, :] -= 2 * ds_
        # dq[:,  1:] += 2 * dd_;  dq[:,  :-1] -= 2 * dd_
        # g_smooth_q = dq.ravel()[free_idx] * lam_smooth / n_par

        ds_ = np.diff(q2d, axis=0)   # shape (ns-1, nd) — s-direction differences
        dd_ = np.diff(q2d, axis=1)   # shape (ns, nd-1) — d-direction differences

        smooth_loss = (lam_smooth_s * np.sum(ds_**2) +
                    lam_smooth_d * np.sum(dd_**2)) / n_par

        # s-direction gradient (axis=0): scale by lam_smooth_s
        dq_s = np.zeros((ns, nd), dtype=np.float64)
        dq_s[1:,  :] += 2 * ds_
        dq_s[:-1, :] -= 2 * ds_

        # d-direction gradient (axis=1): scale by lam_smooth_d
        dq_d = np.zeros((ns, nd), dtype=np.float64)
        dq_d[:,  1:] += 2 * dd_
        dq_d[:,  :-1] -= 2 * dd_

        # Combine with their respective weights BEFORE indexing into free_idx
        g_smooth_q = (dq_s.ravel() * lam_smooth_s +
                    dq_d.ravel() * lam_smooth_d)[free_idx] / n_par

        # ── Drift in q-space: R_drift = (λ_d/K_f) · ||q||^2 ─────────────────────
        # Hessian: (2λ_d/K_f) · I  [constant, PD]
        drift_loss = lam_drift * np.mean(q**2)
        g_drift_q  = 2.0 * lam_drift * q / n_free

        if debug: return smooth_loss, drift_loss

        return float(smooth_loss + drift_loss), (g_smooth_q + g_drift_q)



    # def _hessp_std(self, q, vec, M_free, fixed_contrib, v_opt,
    #             free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift, mu=0.0):

    #     fp     = np.exp(q)
    #     fv     = M_free @ fp + fixed_contrib
    #     n_par  = ns * nd
    #     n_free = len(free_idx)

    #     mu_v    = v_opt.mean()
    #     mu_vf   = (v_opt * fv).mean()
    #     alpha   = mu_v / (mu_vf + 1e-12)
    #     fv_norm = alpha * fv
    #     u       = v_opt * fv_norm
    #     mu_u    = u.mean()
    #     N       = len(u)
    #     sigma   = np.std(u) + 1e-12

    #     g_data      = v_opt * (u - mu_u) / (N * sigma)
    #     grad_data_p = alpha * (
    #         M_free.T @ g_data
    #         - (np.dot(fv_norm, g_data) / (N * mu_v)) * (M_free.T @ v_opt)
    #     )

    #     # ── p-space direction ────────────────────────────────────────────────────
    #     p_vec = fp * vec

    #     # ── Data Hessian-vector product (Gauss-Newton, p-space) ──────────────────
    #     Mv    = M_free @ p_vec
    #     term1 = (alpha**2 / (N * sigma))     * (M_free.T @ (v_opt**2 * Mv))
    #     term2 = -(alpha**2 / (N**2 * sigma)) * (M_free.T @ v_opt) * (v_opt @ Mv)
    #     term3 = -(alpha**2 / (N * sigma**2)) * (M_free.T @ (v_opt * (u - mu_u))) * (g_data @ Mv)
    #     H_data_p_pvec = term1 + term2 + term3

    #     # Convert data term to q-space: H^q z = p⊙(H^p(p⊙z)) + ∇_p L ⊙ p ⊙ z
    #     H_data_q = fp * H_data_p_pvec + grad_data_p * fp * vec

    #     # ── Asymmetric smoothness Hessian-vector product ──────────────────────────
    #     # Expand vec into the full (ns, nd) grid, with fixed knots at 0.
    #     v_full           = np.zeros(n_par, dtype=np.float64)
    #     v_full[free_idx] = vec
    #     v_grid           = v_full.reshape(ns, nd)

    #     # s-direction: accumulate with lam_smooth_s
    #     dvs    = np.diff(v_grid, axis=0)
    #     dv_s   = np.zeros((ns, nd), dtype=np.float64)
    #     dv_s[1:,  :] += 2 * dvs
    #     dv_s[:-1, :] -= 2 * dvs

    #     # d-direction: accumulate with lam_smooth_d
    #     dvd    = np.diff(v_grid, axis=1)
    #     dv_d   = np.zeros((ns, nd), dtype=np.float64)
    #     dv_d[:,  1:] += 2 * dvd
    #     dv_d[:,  :-1] -= 2 * dvd

    #     # Combine with respective weights BEFORE indexing, mirroring _reg_and_grad
    #     H_smooth_q = (dv_s.ravel() * lam_smooth_s +
    #                 dv_d.ravel() * lam_smooth_d)[free_idx] / n_par

    #     H_drift_q = 2.0 * lam_drift / n_free * vec

    #     # μI is the LM shift; zero for Trust-NCG (default mu=0).
    #     H_lm_q = mu * vec

    #     return (H_data_q + H_smooth_q + H_drift_q + H_lm_q).astype(np.float64)
    

    def _hessp_std(self, q, vec, M_free, fixed_contrib, v_opt,
                    free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift, mu=0.0):

        fp  = np.exp(q)
        fv  = M_free @ fp + fixed_contrib

        corr   = v_opt * fv
        mu_u   = corr.mean()
        N      = len(corr)
        sigma  = np.std(corr) + 1e-12

        g_data      = v_opt * (corr - mu_u) / (N * sigma)
        grad_data_p = M_free.T @ g_data

        p_vec = fp * vec
        Mv    = M_free @ p_vec

        term1        = (1.0 / (N * sigma))     * (M_free.T @ (v_opt**2 * Mv))
        term2        = -(1.0 / (N**2 * sigma)) * (M_free.T @ v_opt) * (v_opt @ Mv)
        term3        = -(1.0 / (N * sigma**2)) * (M_free.T @ (v_opt * (corr - mu_u))) * (g_data @ Mv)
        H_data_p_pvec = term1 + term2 + term3

        H_data_q = fp * H_data_p_pvec + grad_data_p * fp * vec

        v_full           = np.zeros(ns * nd, dtype=np.float64)
        v_full[free_idx] = vec
        v_grid           = v_full.reshape(ns, nd)

        dvs  = np.diff(v_grid, axis=0)
        dv_s = np.zeros((ns, nd), dtype=np.float64)
        dv_s[1:,  :] += 2 * dvs
        dv_s[:-1, :] -= 2 * dvs

        dvd  = np.diff(v_grid, axis=1)
        dv_d = np.zeros((ns, nd), dtype=np.float64)
        dv_d[:,  1:] += 2 * dvd
        dv_d[:,  :-1] -= 2 * dvd

        H_smooth_q = (dv_s.ravel() * lam_smooth_s +
                        dv_d.ravel() * lam_smooth_d)[free_idx] / (ns * nd)

        H_drift_q = 2.0 * lam_drift / len(free_idx) * vec
        H_lm_q    = mu * vec

        return (H_data_q + H_smooth_q + H_drift_q + H_lm_q).astype(np.float64)

    def _lm_minimize(self, q0, obj_fn, hessp_fn_factory,
                    max_iter, gtol,
                    mu0=1e-3, mu_min=1e-12, mu_max=1e12):
        """
        Levenberg–Marquardt optimizer for the CV-loss objective.

        Each outer iteration:
        1. Solve the damped linear system (H_approx + μI)δ = −g via CG,
            where H_approx is the Gauss-Newton Hessian plus regularization.
        2. Compute the ratio ρ = (actual reduction) / (predicted reduction).
        3. Accept the step and decrease μ if ρ > 0.25;
            reject the step and increase μ otherwise.

        The predicted reduction from the quadratic model is
            m(δ) = g·δ + ½ δ·(H+μI)δ
        which, because (H+μI)δ = −g when CG converges exactly, equals
        −½ g·δ. With truncated CG the Hessian-vector product (H+μI)δ is
        computed explicitly for an accurate predicted reduction.

        Parameters
        ----------
        q0 : ndarray
            Initial log-space parameters.
        obj_fn : callable
            Returns (loss, grad) in q-space.
        hessp_fn_factory : callable
            hessp_fn_factory(mu) → hessp(q, vec); builds the damped
            Hessian-vector product for a given μ.
        max_iter : int
        gtol : float
            Convergence tolerance on the gradient norm.
        mu0 : float
            Initial damping (default 1e-3; increased automatically if the
            first step is not a descent).
        mu_min, mu_max : float
            Bounds on μ to prevent numerical under/overflow.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            .x       — optimal q
            .fun     — final loss value
            .jac     — final gradient
            .nit     — number of outer iterations
            .success — True if gtol was reached
            .message — termination reason
        """
        from scipy.optimize import OptimizeResult
        from scipy.sparse.linalg import LinearOperator, cg as sparse_cg

        q  = q0.copy()
        n  = len(q)
        mu = mu0
        nu = 2.0                      # damping scale factor on rejection

        f, g = obj_fn(q)
        nit  = 0

        for k in range(max_iter):
            g_norm = np.linalg.norm(g)
            if g_norm < gtol:
                break

            # Inner CG solve: (H_approx + μI) δ = −g
            # Tolerance follows the Eisenstat-Walker rule: tighter near the solution.
            cg_tol     = min(0.5, np.sqrt(g_norm))
            cg_maxiter = max(20, 2 * n)
            hessp_mu   = hessp_fn_factory(mu)

            def matvec(v):
                return hessp_mu(q, v)

            A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            delta, cg_info = sparse_cg(A, -g, rtol=cg_tol, maxiter=cg_maxiter)

            # Predicted reduction from quadratic model
            # pred = −g·δ − ½ δ·(H+μI)δ
            # We reuse the Hessian-vector product rather than recomputing it.
            Hd = hessp_mu(q, delta)
            predicted = -(g @ delta) - 0.5 * (delta @ Hd)

            # Actual reduction
            q_new = q + delta
            f_new, g_new = obj_fn(q_new)
            actual = f - f_new

            nit += 1

            # Step acceptance and μ update
            if predicted <= 0.0:
                # Quadratic model is not decreasing — GN curvature is unreliable.
                # Increase damping without accepting the step.
                mu  = min(mu * nu, mu_max)
                nu *= 2.0
                print(f"  LM iter {nit:3d}  f={f:.6g}  |g|={g_norm:.3g}"
                    f"  ρ=N/A (bad model)  μ={mu:.3g}", flush=True)
                continue

            rho = actual / predicted

            if rho > 0.0:
                # Accept step
                q, f, g = q_new, f_new, g_new
                nu = 2.0                       # reset scale factor on acceptance
                if rho > 0.75:
                    mu = max(mu / 3.0, mu_min)  # step was very good: reduce damping
                elif rho < 0.25:
                    mu = min(mu * nu, mu_max)   # step was mediocre: increase damping
                # 0.25 ≤ ρ ≤ 0.75: keep μ unchanged
            else:
                # Reject step: increase damping and try again
                mu  = min(mu * nu, mu_max)
                nu *= 2.0

            print(f"  LM iter {nit:3d}  f={f:.6g}  |g|={g_norm:.3g}"
                f"  ρ={rho:+.3f}  μ={mu:.3g}"
                f"  {'accepted' if rho > 0 else 'REJECTED'}", flush=True)

        g_norm_final = float(np.linalg.norm(g))
        success      = g_norm_final < gtol
        message      = ('Gradient norm below gtol.' if success
                        else 'Maximum iterations reached.')

        return OptimizeResult(
            x=q, fun=float(f), jac=g,
            nit=nit, success=success, message=message,
        )


    # def _objective_std(self, q, M_free, fixed_contrib, v_opt,
    #                 free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift):
    #     """
    #     Standard deviation objective function for L-BFGS optimization.
    #     """
    #     fp      = np.exp(q)
    #     fv      = M_free @ fp + fixed_contrib

    #     mu_v    = v_opt.mean()
    #     mu_vf   = (v_opt * fv).mean()
    #     alpha   = mu_v / (mu_vf + 1e-12)
    #     fv_norm = alpha * fv

    #     corr  = v_opt * fv_norm
    #     mu    = corr.mean()
    #     nm    = len(corr)
    #     sigma = np.std(corr)

    #     g           = v_opt * (corr - mu) / (nm * sigma + 1e-12)
    #     grad_data_p = alpha * (
    #         M_free.T @ g
    #         - (np.dot(fv_norm, g) / (nm * mu_v)) * (M_free.T @ v_opt)
    #     )
    #     grad_data_q = grad_data_p * fp

    #     reg_loss, g_reg_q = self._reg_and_grad(
    #         q, free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift
    #     )
    #     return float(sigma) + reg_loss, (grad_data_q + g_reg_q).astype(np.float64)

    def _objective_std(self, q, M_free, fixed_contrib, v_opt,
                        free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift):

        fp  = np.exp(q)
        fv  = M_free @ fp + fixed_contrib

        corr  = v_opt * fv
        mu    = corr.mean()
        nm    = len(corr)
        sigma = np.std(corr)

        g           = v_opt * (corr - mu) / (nm * sigma + 1e-12)
        grad_data_p = M_free.T @ g
        grad_data_q = grad_data_p * fp

        reg_loss, g_reg_q = self._reg_and_grad(
            q, free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift
        )
        return float(sigma) + reg_loss, (grad_data_q + g_reg_q).astype(np.float64)

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
        d_profile = bin_grid.mean(axis=0, keepdims=True)
        bin_grid  = np.broadcast_to(d_profile, bin_grid.shape).copy()

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
        lam_smooth_s       = 0.15,
        lam_smooth_d       = 1.5,
        lam_drift          = 0.10,
        field_bounds       = (0.2, 3.0),
        max_iter           = 300,
        ftol               = 1e-9,
        gtol               = 1e-6,
        _q0                = None,
        solver             = 'lbfgsb',
        bc_radius_px       = 0.0,
        seam_radius_px     = 0.0,
        d_bc_thresh        = 0.90,
        lm_mu0             = 1e-3,
        lm_tau             = 0.0,
        **kwargs
    ):
        """
        Fits the (s, d) correction field using L-BFGS-B, Trust-NCG, or
        Levenberg–Marquardt optimization.

        Parameters
        ----------
        solver : str
            'lbfgsb'     — L-BFGS-B with box constraints (any objective).
            'trust-ncg'  — Trust-Region Newton-CG (objective='std' only).
            'lm'         — Levenberg–Marquardt (objective='std' only).
        lm_mu0 : float
            Initial damping parameter μ₀ for LM (default 1e-3).
        lm_tau : float
            If > 0, μ₀ is set to lm_tau · ‖g(q₀)‖ at the start of LM,
            which adapts the initial damping to the gradient scale.
            Set to 0 to use lm_mu0 directly (default).

        All other parameters are unchanged from the original method.
        """
        from scipy.optimize import minimize
        from scipy.interpolate import RectBivariateSpline
        from scipy.ndimage import sobel
        import time

        if objective not in ('std'):
            raise ValueError(f"objective must be 'std' got '{objective}'.")
        if solver not in ('lbfgsb', 'trust-ncg', 'lm'):
            raise ValueError(f"solver must be 'lbfgsb', 'trust-ncg', or 'lm', got '{solver}'.")
        if solver in ('trust-ncg', 'lm') and objective != 'std':
            raise ValueError(f"solver='{solver}' is only supported with objective='std'.")

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
        # Pre-normalize v_opt so that a field of all-ones gives the right mean.
        # This replaces the per-iteration alpha normalization in the objective,
        # making the loss sensitive to the absolute scale of the field.
        v_opt_mean = float(v_opt.mean())
        v_opt = v_opt / (v_opt_mean + 1e-12)
        n_par = ns * nd

        print(f"Building M ({len(v_opt)} pixels × {n_par} params)…", flush=True)
        t0 = time.time()
        M_free, free_idx, fixed_contrib, bc_idx = self._build_interp_matrix(
            sc, dc, s_opt, d_opt, ns, nd
        )
        print(f"  done in {time.time()-t0:.1f}s  ({len(free_idx)} free params)", flush=True)

        self._last_sc = sc
        self._last_dc = dc

        merge_map  = {}
        knot_xy    = self._compute_knot_positions(sc, dc, opt_r, opt_c, s_opt, d_opt)
        self._last_knot_xy = knot_xy

        # Cumulative boolean mask over the original free_idx.
        # Each BC call returns a keep mask relative to the free_idx it received,
        # so we compose them by indexing into the running mask.
        keep_cumulative = np.ones(len(free_idx), dtype=bool)

        if bc_radius_px > 0.0:
            M_free, free_idx, medial_merge_map, medial_keep = \
                self._apply_medial_boundary_conditions(
                    knot_xy, dc, M_free, free_idx,
                    n_par, ns, nd,
                    d_bc_thresh=d_bc_thresh,
                    bc_radius_px=bc_radius_px,
                )
            # medial_keep is relative to the free_idx that entered the medial call,
            # which still corresponds 1-to-1 with keep_cumulative's True positions.
            keep_cumulative[keep_cumulative] = medial_keep
            merge_map.update(medial_merge_map)

        if seam_radius_px > 0.0:
            M_free, free_idx, seam_merge_map, seam_keep = \
                self._apply_seam_boundary_conditions(
                    knot_xy, dc, M_free, free_idx,
                    n_par, ns, nd,
                    seam_radius_px=seam_radius_px,
                )
            # seam_keep is relative to the free_idx that entered the seam call
            # (already reduced by medial BC), so index into the surviving positions.
            keep_cumulative[keep_cumulative] = seam_keep
            merge_map.update(seam_merge_map)

        if _q0 is not None:
            _q0 = _q0[keep_cumulative]

        self._last_merge_map = merge_map
        self._last_free_idx  = free_idx

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

        def obj_fn(q):
            return self._objective_std(q, M_free, fixed_contrib, v_opt,
                                    free_idx, ns, nd, lam_smooth_s, lam_smooth_d, lam_drift)

        print(f"Optimizing [{objective}] with [{solver}]…", flush=True)
        t1 = time.time()

        if solver == 'trust-ncg':
            # Original Trust-NCG path — unchanged.
            def hessp_fn(q, vec):
                return self._hessp_std(q, vec, M_free, fixed_contrib, v_opt,
                                    free_idx, ns, nd, lam_smooth_d, lam_smooth_s, lam_drift, mu=0.0)
            result = minimize(
                obj_fn, q0, jac=True, method='trust-ncg',
                hessp=hessp_fn,
                options={'maxiter': max_iter, 'gtol': gtol}
            )

        elif solver == 'lm':
            # ── Levenberg–Marquardt ───────────────────────────────────────────────
            # Determine initial damping μ₀.
            if lm_tau > 0.0:
                # Scale by the gradient norm at the starting point so that μ₀
                # is commensurate with the actual curvature scale of the problem.
                _, g0 = obj_fn(q0)
                mu0_actual = lm_tau * float(np.linalg.norm(g0))
                print(f"  LM μ₀ = lm_tau · ‖g₀‖ = {lm_tau} · {np.linalg.norm(g0):.3g}"
                    f" = {mu0_actual:.3g}", flush=True)
            else:
                mu0_actual = lm_mu0

            # Factory: returns a damped hessp callable for a given μ.
            def hessp_fn_factory(mu):
                def hessp(q, vec):
                    return self._hessp_std(
                        q, vec, M_free, fixed_contrib, v_opt,
                        free_idx, ns, nd,  lam_smooth_d, lam_smooth_s, lam_drift, mu=mu
                    )
                return hessp

            result = self._lm_minimize(
                q0, obj_fn, hessp_fn_factory,
                max_iter=max_iter,
                gtol=gtol,
                mu0=mu0_actual,
            )

        else:
            # Original L-BFGS-B path
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
        for follower_flat, leader_flat in merge_map.items():
            opt_par[follower_flat] = opt_par[leader_flat]

        spline_opt = RectBivariateSpline(sc, dc, opt_par.reshape(ns, nd), kx=kx, ky=ky)
        s_all = s_norm[mask].astype(np.float64)
        d_all = d_norm[mask].astype(np.float64)

        field = np.ones((rows, cols), dtype=np.float64)
        field[mask] = np.clip(spline_opt.ev(s_all, d_all),
                            field_bounds[0], field_bounds[1])

        return self._apply_field(field), field, result
    
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
                        origin='upper', interpolation='nearest')
        im_f = ax_field.imshow(field_display, cmap=cmap_f, norm=norm_f,
                            origin='upper', interpolation='nearest')
        ax_field.scatter(kc_flat, kr_flat, c=kv_flat, cmap=cmap_f, norm=norm_f,
                        s=30, edgecolors='k', linewidths=0.5, zorder=5)
        cb_f = fig.colorbar(im_f, ax=ax_field, fraction=0.046, pad=0.04)
        cb_f.set_label('f(x,y)', fontsize=9)
        cb_f.locator = MaxNLocator(nbins=5); cb_f.update_ticks()
        ax_field.set_title(f'Field on breast ({ns}x{nd} knots)', fontsize=11)
        ax_field.axis('off')

        # [1,0] Knot grid in image space — edge colour encodes BC group membership
        ax_knots.imshow(img_orig, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                        origin='upper', interpolation='nearest')
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
