import numpy as np
from enum import Enum

class State(Enum):
    V = 0
    THETA = 1
    X = 2
    Y = 3
    PHI = 4


figure_number = 1

def get_rectangle_corners(bottom_left_corner, side_x, side_y):
    corners = np.empty((4, 2))

    x, y = bottom_left_corner
    corners[0, :] = np.array([x, y])
    corners[1, :] = np.array([x + side_x, y])
    corners[2, :] = np.array([x + side_x, y + side_y])
    corners[3, :] = np.array([x, y + side_y])

    return corners

def pixel_to_xy(pixel: tuple, image_size: tuple, meters_per_pixel: float):
    x, y = pixel
    x = (x - image_size[0]/2) * meters_per_pixel
    y = - (y - image_size[1]/2) * meters_per_pixel
    return x, y

def xy_to_pixel(x: float, y: float, image_size: tuple, meters_per_pixel: float):
    px = x / meters_per_pixel + image_size[0]/2
    py = -y / meters_per_pixel + image_size[1]/2
    
    if px < 0 or px > image_size[1] or py < 0 or py > image_size[0]:
        return None
    
    return int(px), int(py)


if __name__ == '__main__':
    image_size_ex = (1000, 2000)
    meters_per_pixel = 0.1
    pixel_ex = (510, 400)

    px_xeq0, py_yeq0 = (500, 1000)
    xeq0, yeq0 = pixel_to_xy((px_xeq0, py_yeq0), image_size_ex, meters_per_pixel)

    px_xeq5, py_yeq5 = (550, 950)
    xeq5, yeq5 = pixel_to_xy((px_xeq5, py_yeq5), image_size_ex, meters_per_pixel)

    print(f"{xeq0}, {yeq0}")
    print(xy_to_pixel(xeq0, yeq0, image_size_ex, meters_per_pixel))
    print(f"{xeq5}, {yeq5}")
    print(xy_to_pixel(xeq5, yeq5, image_size_ex, meters_per_pixel))
    print(xy_to_pixel(0, 0, image_size_ex, meters_per_pixel))

    x_ex, y_ex = pixel_to_xy(pixel_ex, image_size_ex, meters_per_pixel)
    px_ex, py_ex = xy_to_pixel(x_ex, y_ex, image_size_ex, meters_per_pixel)
    print(f"pixel input: {pixel_ex}, pixel output: {(px_ex, py_ex)}")
    x_ex, y_ex = (5, 7)
    px_ex, py_ex = xy_to_pixel(x_ex, y_ex, image_size_ex, meters_per_pixel)
    xout, yout = pixel_to_xy((px_ex, py_ex), image_size_ex, meters_per_pixel)
    print(f"xy input: {(x_ex, y_ex)}, xy output: {(xout, yout)}")
