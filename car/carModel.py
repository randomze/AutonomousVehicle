from __future__ import annotations
import numpy as np


class CarModel:
    """ Model of the car dynamics.
    """

    def __init__(self, constants):
        self.M = constants['M']
        self.L = constants['L']
        self.r_cm = constants['r_cm']
        self.delta_cm = constants['delta_cm']
        self.Izz = constants['Izz']
        self.idle_power = constants['idle_power']
        self.plot_x = []
        self.plot_y = []

    def derivative(self, instant, state, inputs):
        """ Returns the derivative of the state as a function of time, the state and the inputs.
            The state is [v, theta, x, y, phi]. v and phi are the magnitude and direction of the 
            velocity of the car at the front wheel. theta, x and y are the orientation and position
            of the car.
        """
        v, theta, x, y, phi = state
        f_v, omega_s = inputs

        v_dot = (f_v - ((self.M - self.r_cm**2 - self.Izz/self.L**2) * np.cos(phi)
                        - 2 * self.r_cm * np.cos(self.delta_cm)) * np.sin(phi) * omega_s * v) / \
                (self.M * np.cos(phi)**2 + (self.r_cm**2 + self.Izz/self.L**2)*np.sin(phi)**2
                 - 2 * self.r_cm * np.cos(phi) * np.cos(self.delta_cm))
        theta_dot = v * np.sin(phi) / self.L
        x_dot = v * np.cos(theta) * np.cos(phi)
        y_dot = v * np.sin(theta) * np.cos(phi)
        phi_dot = omega_s

        return np.array([v_dot, theta_dot, x_dot, y_dot, phi_dot])

    def output(self, instant, state):
        """ Returns the state of the car, with the velocity of the center of mass instead of the 
            velocity at the front wheel.
        """
        v, theta, x, y, phi = state
        v_cm = v * np.sqrt(np.cos(phi)**2 + self.r_cm**2 * np.sin(phi)**2 - self.r_cm
                           * np.sin(2 * phi) * np.cos(self.delta_cm))
        return np.array([v_cm, theta, x, y, phi])
