import numpy as np
import matplotlib.pyplot as plt

class CarModel:
    def __init__(self, constants):
        self.M = constants['M']
        self.L = constants['L']
        self.r_cm = constants['r_cm']
        self.delta_cm = constants['delta_cm']
        self.Izz = constants['Izz']
        self.Izz_phi = constants['Izz_phi']
        self.idle_power = constants['idle_power']
        self.plot_x = []
        self.plot_y = []
        
    def derivative(self, instant, state, inputs):
        v, theta, x, y, phi, phi_dot = state
        f_v, tau_phi = inputs

        v_dot = (f_v -((self.M - self.r_cm**2 - self.Izz/self.L**2) * np.cos(phi) \
                - 2 * self.r_cm * np.cos(self.delta_cm)) * np.sin(phi) * phi_dot * v) / \
                (self.M * np.cos(phi)**2 + (self.r_cm**2 + self.Izz/self.L**2)*np.sin(phi)**2 \
                - 2 * self.r_cm * np.cos(phi) * np.cos(self.delta_cm))
        theta_dot = v * np.sin(phi) / self.L
        x_dot = v * np.cos(theta) * np.cos(phi)
        y_dot = v * np.sin(theta) * np.cos(phi)
        phi_dot_dot = (tau_phi + 1/2*(self.r_cm**2 * self.M + self.Izz/self.L**2 - self.M) * np.sin(2*phi) -
                        2*self.r_cm*np.cos(2*phi)*np.cos(self.delta_cm)) / (self.Izz_phi)

        return np.array([v_dot, theta_dot, x_dot, y_dot, phi_dot, phi_dot_dot])

    def output(self, instant, state):
        v, theta, x, y, phi, phi_dot = state
        v_cm = v * np.sqrt(np.cos(phi)**2 + self.r_cm**2 * np.sin(phi)**2 - self.r_cm\
                             * np.sin(2 * phi) * np.cos(self.delta_cm))
        return np.array([v_cm, theta, x, y, phi, phi_dot])
