import numpy as np
import matplotlib.pyplot as plt

class CarModel:
    def __init__(self, constants):
        self.M = constants['M']
        self.L = constants['L']
        self.r_cm = constants['r_cm']
        self.delta_cm = constants['delta_cm']
        self.Izz = constants['Izz']
        self.plot_x = []
        self.plot_y = []
        
    def derivative(self, instant, state, inputs):
        v, theta, x, y, phi = state
        f_v, omega_s = inputs

        v_dot = (f_v -((self.M - self.r_cm**2 - self.Izz/self.L**2) * np.cos(phi) \
                - 2 * self.r_cm * np.cos(self.delta_cm)) * np.sin(phi) * omega_s * v) / \
                (self.M * np.cos(phi)**2 + (self.r_cm**2 + self.Izz/self.L**2)*np.sin(phi)**2 \
                - 2 * self.r_cm * np.cos(phi) * np.cos(self.delta_cm))
        theta_dot = v * np.sin(phi) / self.L
        x_dot = v * np.cos(theta) * np.cos(phi)
        y_dot = v * np.sin(theta) * np.cos(phi)
        phi_dot = omega_s

        # Saturate phi in order not to reach weird stuff

        return np.array([v_dot, theta_dot, x_dot, y_dot, phi_dot])

    def output(self, instant, state):
        return state

    def plot(self, instant, state, fig_number):
        plt.figure(fig_number)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

        _, _, x, y, _ = state
        self.plot_x += [x]
        self.plot_y += [y]

        plt.scatter(x, y, c='r')
        plt.show()
        plt.pause(0.01)