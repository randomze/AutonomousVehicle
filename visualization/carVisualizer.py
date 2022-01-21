import matplotlib.pyplot as plt
import numpy as np
from .utils import get_rectangle_corners, figure_number

class CarVisualizer:
    def __init__(self, car_constants):
        self.L = car_constants['L']
        self.Lr = car_constants['Lr']
        self.Lf = car_constants['Lf']
        self.d = car_constants['d']
        self.r = car_constants['r']
        self.wheel_width = car_constants['wheel_width']

        self.car_length = self.L + self.Lr + self.Lf
        self.car_width = 2 * self.d

    def plot(self, state):
        plt.figure(figure_number)
        plt.clf()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

        car_rectangles = self.get_car_representation(state)

        for i in range(4):
            rectangle = car_rectangles[i*4:i*4+4,:]
            plt.plot(np.append(rectangle[:, 0], rectangle[0, 0]), np.append(rectangle[:, 1], rectangle[0, 1]))

        plt.show()
        plt.pause(0.01)

    def get_car_representation(self, state):
        v, theta, x, y, phi = state
        car_body = get_rectangle_corners([-self.Lr, -self.d], self.car_length, self.car_width)

        front_wheel_rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], \
                                                [np.sin(phi), np.cos(phi)]])
        front_wheel_translation = np.array([[self.L], [0]])
        front_wheel = get_rectangle_corners([-self.r, -self.wheel_width/2], 2*self.r, self.wheel_width)
        front_wheel = np.transpose(front_wheel_rotation_matrix @ front_wheel.T + front_wheel_translation)

        left_rear_wheel = get_rectangle_corners([-self.r, -self.d - self.wheel_width], 2*self.r, self.wheel_width)
        right_rear_wheel = get_rectangle_corners([-self.r, self.d], 2*self.r, self.wheel_width)

        car_rectangles = np.block([[car_body], [front_wheel], [left_rear_wheel], [right_rear_wheel]])
        car_rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], \
                                         [np.sin(theta), np.cos(theta)]])
        car_translation = np.array([[x], [y]])
        car_rectangles = np.transpose(car_rotation_matrix @ car_rectangles.T + car_translation)

        return car_rectangles
