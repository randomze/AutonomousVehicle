from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from path.utils import get_rectangle_corners

# The CarVisualizer object deals with displaying the car in the plots and video.
class CarVisualizer:
    def __init__(self, car_constants):
        ''' Initialize the car constants from the input parameters.
        '''
        self.L = car_constants['L']
        self.Lr = car_constants['Lr']
        self.Lf = car_constants['Lf']
        self.d = car_constants['d']
        self.r = car_constants['r']
        self.wheel_width = car_constants['wheel_width']

        self.car_length = self.L + self.Lr + self.Lf
        self.car_width = 2 * self.d
        self.frame = 0

        self.state = None
        self.lines = []

    def set_state(self, state):
        ''' State setter.
        '''
        self.state = state

    def plot(self, ax: plt.Axes, zorder=1):
        ''' Plot the car in it's current state.
        '''
        # Get the car rectangles (body, rear wheels, front wheel)
        car_rectangles = self.get_car_representation(self.state)
        rectangles = []
        # If no lines have yet been plotted, plot the initial state of the car from it's rectangles.
        if len(self.lines) == 0:
            for i in range(4):
                rectangle = car_rectangles[i*4:i*4+4,:]
                rectangles.append(rectangle)
            rect_body = ax.add_patch(plt.Polygon(rectangles[0], closed=True, fill=True, edgecolor='#1f77b4', facecolor='#1f77b4', zorder=zorder))
            self.lines.append(rect_body)
            rect_front_wheel = ax.add_patch(plt.Polygon(rectangles[1], closed=True, fill=True, edgecolor='C1', facecolor='C1', zorder=zorder))
            self.lines.append(rect_front_wheel)
            rect_left_rear_wheel = ax.add_patch(plt.Polygon(rectangles[2], closed=True, fill=True, edgecolor='C1', facecolor='C1', zorder=zorder))
            self.lines.append(rect_left_rear_wheel)
            rect_right_rear_wheel = ax.add_patch(plt.Polygon(rectangles[3], closed=True, fill=True, edgecolor='C1', facecolor='C1', zorder=zorder))
            self.lines.append(rect_right_rear_wheel)
            return
        # Otherwise, simply update the lines.
        for i in range(4):
            rectangle = car_rectangles[i*4:i*4+4,:]
            self.lines[i].set_xy(rectangle)


    def get_car_representation(self, state):
        ''' Constructs the car's rectangles from the car's current state.
        '''
        # Unpack the state variables
        v, theta, x, y, phi = state
        # Construct the car rectangle in it's own frame
        car_body = get_rectangle_corners([-self.Lr, -self.d], self.car_length, self.car_width)

        # Construct the front wheel in it's own frame and rotate it into the car frame using the
        # steering wheel's orientation and move it into the right position in the car frame
        front_wheel_rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], \
                                                [np.sin(phi), np.cos(phi)]])
        front_wheel_translation = np.array([[self.L], [0]])
        front_wheel = get_rectangle_corners([-self.r, -self.wheel_width/2], 2*self.r, self.wheel_width)
        front_wheel = np.transpose(front_wheel_rotation_matrix @ front_wheel.T + front_wheel_translation)

        # Construct the real wheels in the car frame
        left_rear_wheel = get_rectangle_corners([-self.r, -self.d - self.wheel_width], 2*self.r, self.wheel_width)
        right_rear_wheel = get_rectangle_corners([-self.r, self.d], 2*self.r, self.wheel_width)

        # Rotate all the car's rectangles into the car's orientation in the world frame and move
        # them to the correct position
        car_rectangles = np.block([[car_body], [front_wheel], [left_rear_wheel], [right_rear_wheel]])
        car_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], \
                                         [np.sin(theta), np.cos(theta)]])
        car_translation = np.array([[x], [y]])
        car_rectangles = np.transpose(car_rotation_matrix @ car_rectangles.T + car_translation)

        return car_rectangles
