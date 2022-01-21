import numpy as np

figure_number = 1

def get_rectangle_corners(bottom_left_corner, side_x, side_y):
    corners = np.empty((4, 2))

    x, y = bottom_left_corner
    corners[0, :] = np.array([x, y])
    corners[1, :] = np.array([x + side_x, y])
    corners[2, :] = np.array([x + side_x, y + side_y])
    corners[3, :] = np.array([x, y + side_y])

    return corners