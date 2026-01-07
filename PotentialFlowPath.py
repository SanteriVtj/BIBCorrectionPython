import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, UnivariateSpline
from grid import create_initial_grid

def generate_breast_paths(start_points, img_width, num_steps=100):
    """
    Generates non-crossing paths from boundary points to the right edge.
    
    start_points: List of (y, x) coordinates of the red dots
    img_width: The x-coordinate of the right boundary
    """
    # 1. Sort points by Y to ensure order-preservation (prevents crossing)
    sorted_points = sorted(start_points, key=lambda p: p[0])
    
    paths = []
    
    # 2. Define the target Y coordinates on the right boundary (chest wall)
    # To maintain geometry, we map the Y-range of the start points 
    # to a similar range on the right edge.
    y_starts = [p[0] for p in sorted_points]
    y_ends = np.linspace(min(y_starts), max(y_starts), len(y_starts))
    
    for i, (y_start, x_start) in enumerate(sorted_points):
        y_target = y_ends[i]
        
        # Create a smooth x-trajectory from the skin line to the edge
        x_coords = np.linspace(x_start, img_width - 1, num_steps)
        
        # 3. Use a blending function (Sigmoid or Cubic) to transition Y
        # This ensures the path starts at the red dot and ends at the chest wall
        # smoothly without overlapping neighbors.
        t = (x_coords - x_start) / (img_width - x_start)
        # Smoothstep blending: 3t^2 - 2t^3
        blend = 3*t**2 - 2*t**3
        y_coords = y_start + (y_target - y_start) * blend
        
        paths.append((x_coords, y_coords))
        
    return paths

def interpolate_to_grid(image, paths, grid_width=100, grid_height=200):
    """
    Interpolates image data onto a structured grid defined by the yellow paths.
    """
    # 1. Create the destination 'rectified' coordinate system
    # u: coordinate along the path (0 to 1, from skin to chest wall)
    # v: coordinate across paths (0 to 1, from top path to bottom path)
    u_vec = np.linspace(0, 1, grid_width)
    v_vec = np.linspace(0, 1, grid_height)
    u_grid, v_grid = np.meshgrid(u_vec, v_vec)

    # 2. Collect all (x, y) coordinates from your paths to build the mapping
    path_coords_x = []
    path_coords_y = []
    values = []

    for i, (px, py) in enumerate(paths):
        path_coords_x.extend(px)
        path_coords_y.extend(py)
        
        # Sample the image intensity at these coordinates
        # Using simple rounding for indexing; for better quality use map_coordinates
        for x, y in zip(px, py):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]:
                values.append(image[iy, ix])
            else:
                values.append(0)

    # 3. Define the source points for the interpolation
    # We treat each point on a path as (normalized_path_index, normalized_step_index)
    points = []
    for path_idx in range(len(paths)):
        for step_idx in range(len(paths[0][0])):
            points.append((step_idx / (len(paths[0][0]) - 1), path_idx / (len(paths) - 1)))

    # 4. Perform the interpolation to get the rectified image
    rectified_image = griddata(points, values, (u_grid, v_grid), method='cubic')
    
    return rectified_image

def get_boundary_normals(points):
    """
    Calculates the normal vectors for a set of boundary points.
    Points should be ordered along the boundary.
    """
    normals = []
    for i in range(len(points)):
        # Use neighbors to find the tangent (finite difference)
        p_prev = points[max(0, i-1)]
        p_next = points[min(len(points)-1, i+1)]
        
        # Tangent vector (dx, dy)
        tangent = np.array([p_next[1] - p_prev[1], p_next[0] - p_prev[0]])
        
        # Normal vector: rotate tangent 90 degrees and normalize
        # Assuming the breast is on the left, normal should point right
        normal = np.array([tangent[1], -tangent[0]]) 
        norm = np.linalg.norm(normal)
        normals.append(normal / norm if norm != 0 else np.array([1, 0]))
        
    return normals

def generate_normal_paths(start_points, img_width, num_steps=100):
    normals = get_boundary_normals(start_points)
    paths = []
    
    # Define destination Y coordinates to keep paths from crossing
    y_starts = [p[0] for p in start_points]
    y_ends = np.linspace(min(y_starts), max(y_starts), len(y_starts))
    
    for i, (p_start, n) in enumerate(zip(start_points, normals)):
        y_start, x_start = p_start
        y_target = y_ends[i]
        x_target = img_width
        
        # Cubic Bézier Control Points
        # P0: Start, P3: End
        # P1: P0 + (normal * weight) to force the starting direction
        # P2: A point near the end to level the path out
        weight = (x_target - x_start) * 0.4 # Adjust weight for "straightness"
        
        p0 = np.array([x_start, y_start])
        p1 = p0 + n * weight
        p2 = np.array([x_target - weight, y_target])
        p3 = np.array([x_target, y_target])
        
        # Calculate Bézier path
        t = np.linspace(0, 1, num_steps)
        # B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        path = (1-t)[:,None]**3 * p0 + \
               3*(1-t)[:,None]**2 * t[:,None] * p1 + \
               3*(1-t)[:,None] * t[:,None]**2 * p2 + \
               t[:,None]**3 * p3
               
        paths.append((path[:, 0], path[:, 1]))
        
    return paths

def generate_pixel_perfect_paths(start_points, img_width):
    """
    Generates paths that visit every discrete pixel from skin to chest wall
    starting in the direction of the boundary normal.
    """
    # 1. Calculate normals for the starting points
    normals = get_boundary_normals(start_points)

    discrete_paths = []
    y_ends = np.linspace(min(p[0] for p in start_points), 
                         max(p[0] for p in start_points), 
                         len(start_points))

    for i, (p_start, n) in enumerate(zip(start_points, normals)):
        y_s, x_s = p_start
        y_e, x_e = y_ends[i], img_width - 1
        
        # Control points for the Bézier curve
        dist = x_e - x_s
        p0 = np.array([x_s, y_s])
        p1 = p0 + n * (dist * 0.4) # Push along the normal
        p2 = np.array([x_e - (dist * 0.2), y_e])
        p3 = np.array([x_e, y_e])

        # 2. High-density sampling to ensure no pixels are skipped
        # We sample 10x the width of the image
        t_samples = np.linspace(0, 1, int(dist * 10))
        path_continuous = (1-t_samples)[:,None]**3 * p0 + \
                          3*(1-t_samples)[:,None]**2 * t_samples[:,None] * p1 + \
                          3*(1-t_samples)[:,None] * t_samples[:,None]**2 * p2 + \
                          t_samples[:,None]**3 * p3
        
        # 3. Discretize to pixel coordinates
        path_pixels = np.round(path_continuous).astype(int)
        
        # 4. Remove consecutive duplicates to create a clean 'chain' of pixels
        # This ensures the path is one-pixel-wide and connected
        unique_mask = np.diff(path_pixels, axis=0, prepend=0).any(axis=1)
        unique_path = path_pixels[unique_mask]
        
        discrete_paths.append(unique_path)
        
    return discrete_paths

def correction_field(M, paths, R, C):
    # 1. Calculate the average profile across all paths
    # M is our rectified matrix (rows = paths, cols = steps from skin to chest)
    average_profile = np.median(M, axis=0) 

    # 2. Define the Target Uniformity
    # We assume the 'true' tissue intensity is represented by the stable 
    # region further away from the hyperintense border.
    target_intensity = np.percentile(average_profile, 50) 

    # 3. Compute the raw correction factors
    # C = Target / Observed
    raw_correction = target_intensity / average_profile

    # 4. Apply Spline Smoothing
    # "The polynomial order should be carefully selected" to avoid overcorrecting.
    # A low-order spline (e.g., k=3) ensures we don't dampen lesions.
    steps = np.arange(len(raw_correction))
    spline = UnivariateSpline(steps, raw_correction, k=3, s=1.0)
    smooth_correction_1d = spline(steps)

    # Collect all (x, y) coordinates from your paths and their corresponding correction values
    points = []
    values = []

    for i, (path_x, path_y) in enumerate(paths):
        for j in range(len(path_x)):
            points.append((path_x[j], path_y[j]))
            values.append(smooth_correction_1d[j])

    # Interpolate to create the full 2D correction field C(x, y)
    grid_y, grid_x = np.mgrid[0:R, 0:C]
    correction_field_2d = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=1.0)
    return correction_field_2d

def uniform_spaced_paths(img, nx=10, ny=10):
    # Find starting points for the path
    boundary_nodes = create_initial_grid(img,ny)

    # Generate paths that are directed along the boundary normal
    paths = generate_normal_paths([(y, x) for x,y in boundary_nodes[1:-1,:]], img.shape[1]-1, num_steps=nx)
    paths = np.int64(paths)

    # Create rectified matrix
    M = np.array([[img[x,y] for y, x in zip(*np.int64(path))] for path in paths])

    correction_field_2d = correction_field(M, paths, img.shape[0], img.shape[1])

    # Apply the correction: I_corrected = I_RC * C
    corrected_image = img * correction_field_2d

    return corrected_image, correction_field_2d

def distance_weighted_paths(img, nx=10, ny=10, height_weight=100):
    # Find starting points for the path
    boundary_nodes = create_initial_grid(img,ny)

    # Generate paths that are directed along the boundary normal
    paths = generate_pixel_perfect_paths([(y, x) for x,y in boundary_nodes[1:-1,:]], img.shape[1]-1)
    paths =  [(path[:,0], path[:,1]) for path in paths]
    
    # 3d coordinates for the paths
    paths3d = [[np.array([y,x,height_weight*img[x,y]]) for y, x in zip(*np.int64(path))] for path in paths]

    # Sample paths uniformly in the path and project back to x-y-plane
    path_points = []
    for path in paths3d:
        d = 0
        distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))/10
        sampled_path = []
        sampled_path.append(np.array([path[0][0],path[0][1]]))
        for i in range(len(path)-1):
            y,x,z = path[i]
            y_prev,x_prev,z_prev = path[i-1]
            if d >= distance:
                sampled_path.append(np.array([y,x]))
                d = 0
            d += np.linalg.norm(np.array([y,x,z]-np.array([y_prev,x_prev,z_prev])))
        sampled_path.append(np.array([path[-1][0],path[-1][1]]))
        path_points.append(sampled_path)

    # Create rectified matrix
    M = np.array([[img[x,y] for y, x in np.int64(path)] for path in path_points])

    correction_field_2d = correction_field(M, paths, img.shape[0], img.shape[1])

    # Apply the correction: I_corrected = I_RC * C
    corrected_image = img * correction_field_2d

    return corrected_image, correction_field_2d