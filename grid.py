import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from pydicom import dcmread

def create_initial_grid(img, n_nodes):
    R,C = img.shape
    # find nonzeros in each row
    firsts_r = np.argmax(img>0, axis=1)
    # find first row with non-zero element 
    first_row = np.argmax(firsts_r>0)
    # find last row with non-zero element
    last_row = R-np.argmax(firsts_r[::-1]>0)-1
    # find the corresponding columns
    first_rc = np.argmax(img[first_row,:]>0)

    # Initialize the first node and a grid for boundary coordinates
    p_prev = np.array([first_rc,first_row])
    coords = np.zeros((last_row-first_row,2),np.int64)
    dist = 0
    for y in range(last_row-first_row):
        x = np.argmax(img[first_row+y,:]>0)
        p = np.array([x,first_row+y])
        dist += np.linalg.norm(p_prev-p)
        coords[y,:] = p_prev
        p_prev = p
    coords[-1,:] = p

    # Initialize an array for nodes and set first and last
    node_coords = np.zeros((n_nodes,2))
    step_size = dist/(n_nodes-1)
    node_coords[0,:] = coords[0,:]
    node_coords[-1,:] = coords[-1,:]
    node_i = 0

    # Find node coordinates from the boundary according to the step size
    for coord in coords:
        if np.linalg.norm(coord-node_coords[node_i,:])<step_size:
            continue
        node_coords[node_i+1,:] = coord
        node_i += 1
        if node_i == n_nodes:
            break

    return node_coords