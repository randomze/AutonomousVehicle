import numpy as np

class Sensors:
    def __init__(self, parameters):
        pass

    def output(self, instant, input):
        v, theta, x, y, phi = input
        return np.array([x, y])