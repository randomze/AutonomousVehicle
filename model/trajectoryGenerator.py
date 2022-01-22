import numpy as np

class TrajectoryGenerator:

    def __init__(self):
        pass

    def output(self, instant):
        if instant < 20:
            return np.array([0, np.pi/2, 10, 10, 0])
        elif instant < 30:
            return np.array([0, np.pi/2, 10, 20, 0])
        else:
            return np.array([0, np.pi/2, 10, 30, 0])