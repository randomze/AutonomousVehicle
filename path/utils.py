from __future__ import annotations

import numpy as np

# Get rectangle corner coordinates knowing the rectangles bottom left corner
# and side lengths, assuming it is not angled.
def get_rectangle_corners(bottom_left_corner, side_x, side_y):
    corners = np.empty((4, 2))

    x, y = bottom_left_corner
    corners[0, :] = np.array([x, y])
    corners[1, :] = np.array([x + side_x, y])
    corners[2, :] = np.array([x + side_x, y + side_y])
    corners[3, :] = np.array([x, y + side_y])

    return corners

# Convert pixel coordinates in road image to world coordinates
def pixel_to_xy(pixel: tuple, image_size: tuple, meters_per_pixel: float):
    x, y = pixel
    x = (x - image_size[0]/2) * meters_per_pixel
    y = - (y - image_size[1]/2) * meters_per_pixel
    return x, y

# Convert world coordinates to pixel coordinates in road image
def xy_to_pixel(x: float, y: float, image_size: tuple, meters_per_pixel: float):
    px = x / meters_per_pixel + image_size[0]/2
    py = -y / meters_per_pixel + image_size[1]/2
    
    if px < 0 or px > image_size[1] or py < 0 or py > image_size[0]:
        return None
    
    return int(px), int(py)


if __name__ == '__main__': # basic tests
    image_size_ex = (1000, 2000)
    meters_per_pixel = 0.1
    pixel_ex = (510, 400)

    x_ex, y_ex = pixel_to_xy(pixel_ex, image_size_ex, meters_per_pixel)
    px_ex, py_ex = xy_to_pixel(x_ex, y_ex, image_size_ex, meters_per_pixel)
    print(f"pixel input: {pixel_ex}, pixel output: {(px_ex, py_ex)}")

    x_ex, y_ex = (5, 7)
    px_ex, py_ex = xy_to_pixel(x_ex, y_ex, image_size_ex, meters_per_pixel)
    xout, yout = pixel_to_xy((px_ex, py_ex), image_size_ex, meters_per_pixel)
    print(f"xy input: {(x_ex, y_ex)}, xy output: {(xout, yout)}")
