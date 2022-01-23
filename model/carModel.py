import numpy as np
import matplotlib.pyplot as plt

class CarModel:
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

    def derivative_jacobian(self, instant, state, inputs):
        v, theta, x, y, phi = state
        f_v, omega_s = inputs

        k_1 = self.M - self.r_cm**2 - self.Izz/self.L**2
        k_2 = 2 * self.r_cm * np.cos(self.delta_cm)
        k_3 = self.r_cm**2 + self.Izz/self.L**2
        k_4 = k_2
        M = self.M
        gradient_f1 = np.array([[- ((k_1 * np.cos(phi) - k_2) * np.sin(phi) * omega_s) / (M * np.cos(phi)**2 +
                                 k_3 * np.sin(phi)**2 - k_4 * np.cos(phi)), 
                                 0,
                                 0,
                                 0,
                                 -(((k_1 * np.cos(phi) - k_2) * np.cos(phi) + k_1 * np.sin(phi)) * omega_s * v *
                                 (M * np.cos(phi)**2 + k_3 * np.sin(phi)**2 - k_4 * np.cos(phi)) + (f_v - 
                                 (k_1 * np.cos(phi) - k_2) * np.sin(phi) * omega_s * v) * (-2 * M * np.sin(phi) * np.cos(phi) +
                                 2 * k_3 * np.cos(phi) * np.sin(phi) + k_4 * np.sin(phi))) / (M * np.cos(phi)**2 + k_3 * np.sin(phi)**2
                                 - k_4 * np.cos(phi)) ** 2,
                                 1 / (M * np.cos(phi) ** 2 + k_3 * np.sin(phi)**2 - k_4 * np.cos(phi)),
                                 -(k_1 * np.cos(phi) - k_2) * np.sin(phi) * omega_s * v / 
                                 (M * np.cos(phi) ** 2 + k_3 * np.sin(phi)**2 - k_4 * np.cos(phi))
                                 ]])
        gradient_f2 = np.array([[np.sin(phi)/self.L,
                                 0,
                                 0,
                                 0,
                                 v*np.cos(phi)/self.L,
                                 0,
                                 0]])
        gradient_f3 = np.array([[np.cos(phi + theta),
                                 - v * np.sin(phi + theta),
                                 0,
                                 0,
                                 - v * np.sin(phi + theta),
                                 0,
                                 0]])
        gradient_f4 = np.array([[np.sin(phi + theta),
                                 v * np.cos(phi + theta),
                                 0,
                                 0,
                                 v * np.cos(phi + theta),
                                 0,
                                 0]])
        gradient_f5 = np.array([[0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 1]])
                            
        jacobian = np.vstack([gradient_f1, gradient_f2, gradient_f3, gradient_f4, gradient_f5])
        return jacobian

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
