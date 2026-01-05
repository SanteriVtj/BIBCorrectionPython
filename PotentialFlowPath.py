import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

# Usage Example:
# rectified = interpolate_to_grid(subtraction_image, paths)

# Example usage with dummy data representing the red dots
# red_dots = [(500, 2300), (800, 1850), (1150, 1500), (1600, 1250), 
#             (2150, 1250), (2600, 1500), (2900, 1950), (3100, 2450)]
# width = 2800

# paths = generate_breast_paths(red_dots, width)

# # Visualization
# plt.figure(figsize=(8, 10))
# for x_p, y_p in paths:
#     plt.plot(x_p, y_p, color='yellow', linewidth=2)
#     plt.scatter(x_p[0], y_p[0], color='red', zorder=5) # Red dots

# plt.gca().invert_yaxis()
# plt.title("Generated Non-Crossing Correction Paths")
# plt.show()