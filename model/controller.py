import numpy as np

class Controller:

    def __init__(self):
        pass

    def output(self, instant, input):
        sensors_output, trajectory_output = input

        return np.array([10, 0.1])