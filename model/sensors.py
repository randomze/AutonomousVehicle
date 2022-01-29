from __future__ import annotations
import numpy as np

class Sensors:
    def __init__(self, parameters):
        self.deadzone_velocity: float = parameters['deadzone_velocity']

    def velocity_deadzone(self, velocity):
        if abs(velocity) < self.deadzone_velocity:
            return 0
        else:
            return velocity - self.deadzone_velocity * np.sign(velocity)

    def output(self, instant: float, input: np.ndarray):
        output = input
        output[0] = self.velocity_deadzone(output[0])
        return output