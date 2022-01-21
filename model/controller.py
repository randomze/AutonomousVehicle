from distutils.log import error
from re import S
import numpy as np
import scipy.linalg
import scipy.optimize

class Controller:

    def __init__(self, car):
        self.previous_error = np.zeros((3,))
        self.admissible_forces = np.logspace(1, 5, 13)
        self.admissible_steers = np.linspace(-np.pi/4, np.pi/4, 9)
        self.input_horizon = 4
        self.output_horizon = 50
        self.car = car

    def output(self, instant, input):
        sensors_output, trajectory_output = input

        current_error = trajectory_output - sensors_output
        heading = sensors_output[1]
        velocity = sensors_output[0]
        current_error = current_error[1:4]

        body_frame_rotation = np.array([[1, 0, 0],
                                        [0, np.cos(heading), np.sin(heading)],
                                        [0, -np.sin(heading), np.cos(heading)]])
        error_body_frame = body_frame_rotation @ current_error

        velocity_apply = 0.16 * error_body_frame[1]
        force_apply = 1000 * (velocity_apply - velocity)
        steering_apply = 10 * error_body_frame[2] + 0.1 * error_body_frame[0]

        return np.array([force_apply, steering_apply])
        # Here lies my attempt at MPC
        horizon = 50
        trajectory_output = np.append(trajectory_output, np.array([0, 0]))
        trajectory = np.tile(trajectory_output, (horizon,))

        initial_variable = np.append(sensors_output, np.array([0, 0]))
        optimization_variable = np.tile(initial_variable, (horizon,))
        big_lambda = np.zeros((horizon * 5,))

        def big_f(big_variables):
            big_f = np.empty((horizon * 5,))
            big_f[:5] = sensors_output - big_variables[:5]
            for i in range(1, horizon):
                big_f[i*5:(i+1)*5] = 0.01 * self.car.derivative(0, big_variables[i*7:i*7 + 5],
                                                              big_variables[i*7 + 5:i*7 + 7]) - big_variables[i*7: i*7 + 5]
            return big_f

        def derivative_big_f(big_variables):
            derivatives = []
            for i in range(1, horizon):
                derivatives += [0.01 * self.car.derivative_jacobian(0, big_variables[i*7:i*7 + 5],
                                                             big_variables[i*7 + 5:i*7 + 7])]

            derivatives = scipy.linalg.block_diag(*derivatives)
            custom_block = np.block([-np.eye(5), np.zeros((5, 2))])
            derivatives = np.block([[custom_block, np.zeros((5, (horizon-1)*7))],
                                    [derivatives, np.zeros(((horizon-1)*5, 7))]])
            return derivatives

        mu = 1e2
        def optimization_function(big_variables):
            return (big_variables - trajectory) + (mu * big_f(big_variables).T + big_lambda.T) @ derivative_big_f(big_variables)

        tol = 1
        first = True
        #while first or np.linalg.norm(big_f(optimization_variable)) > tol:
        optimization_variable = scipy.optimize.fsolve(optimization_function, optimization_variable)
        big_lambda = big_lambda + mu * big_f(optimization_variable)
        first = False

        return optimization_variable[5:7]
        


