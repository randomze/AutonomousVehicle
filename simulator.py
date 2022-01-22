from typing import Tuple
from model.carModel import CarModel
from model.controller import Controller
from model.sensors import Sensors
from model.trajectoryGenerator import TrajectoryGenerator

import matplotlib.pyplot as plt
import numpy as np

from visualization.carVisualizer import CarVisualizer
from visualization.mapVisualizer import MapVisualizer

class Simulator:
    def __init__(self, step_size, car_constants, map_constants, sensorParameters, path: tuple):
        self.step_size = step_size

        self.car_model = CarModel(car_constants)
        self.controller = Controller(self.car_model)
        self.sensors = Sensors(sensorParameters)
        self.trajectory_generator = TrajectoryGenerator(map_constants, path)
        self.car_visualizer = CarVisualizer(car_constants)
        self.map_visualizer = MapVisualizer(map_constants)

    def simulate(self, initial_conditions, final_time, vis_window = ((-20, 20), (-20, 20))):
        car_state = initial_conditions['car_ic']
        controller_output = np.array([0, 0])
        sensors_output = np.array([0, 0])
        trajectory_output = np.array([0, 0])
        
        for instant in np.arange(final_time, step=self.step_size):
            car_input = controller_output
            car_derivative = self.car_model.derivative(instant, car_state, car_input)
            car_state = car_state + car_derivative * self.step_size
            
            if car_state[4] < -np.pi/3:
                car_state[4] = -np.pi/3

            if car_state[4] > np.pi/3:
                car_state[4] = np.pi/3

            car_output = self.car_model.output(instant, car_state)

            sensors_input = car_output
            sensors_output = self.sensors.output(instant, sensors_input)

            trajectory_output = self.trajectory_generator.output(instant)
            controller_input = [sensors_output, trajectory_output]
            controller_output = self.controller.output(instant, controller_input)
            # Do some plots
            self.map_visualizer.plot(car_state, clf=True, window=vis_window)
            self.trajectory_generator.plot()
            self.car_visualizer.plot(car_state, window=vis_window)

def CoM_position(m: int, n: int) -> Tuple:
    d = 0.64
    W = 2 * d
    Lm = 2.2
    Lr = Lf = 0.566
    L = Lr + Lm + Lf

    com_x = 0
    for j in range(n):
        com_x += W/2 - (j - 1/2)*W/n
    com_x = com_x / n

    com_y = 0
    for i in range(m):
        com_y += L/2 - (i - 1/2)*L/m
    com_y = com_y / m

    com_r = np.sqrt((com_x/Lm) ** 2 + (com_y/Lm)**2)
    com_delta = np.arctan2(com_x, com_y)

    return (com_r, com_delta)

def MoI(m: int, n: int) -> float:
    d = 0.64
    W = 2 * d
    Lm = 2.2
    Lr = Lf = 0.566
    L = Lr + Lm + Lf
    M = 810

    Izz = 0
    for i in range(m):
        for j in range(n):
            Izz += (W/2 - (j - 1/2)*W/n)**2 + (L/2 - (i - 1/2)*L/m)**2

    Izz *= M/(m*n)

    return Izz

if __name__ == "__main__":
    road_constants = {
        'lat': 38.7367256,
        'lon': -9.1388871,
        'zoom': 16,
        'upsampling': 3,
        'regularization': 20,
    }
    
    posi = (-5, 10)
    posf = (-7, 30)

    plt.ion()
    m = 3
    n = 2
    com_r, com_delta = CoM_position(m, n)
    Izz = MoI(m, n)
    car_constants = {
        'L': 2.2,
        'Lr': 0.566,
        'Lf': 0.566,
        'd': 0.64,
        'r': 0.256,
        'Length': 3.332,
        'Width': 1.508,
        'M': 800.0,
        'Izz': Izz,
        'r_cm': com_r,
        'delta_cm': com_delta,
        'wheel_width': 0.1
    }  
    sim = Simulator(0.1, car_constants, road_constants, None, (posi, posf))

    initial_conditions = {
        'car_ic': np.array([0, 0, posi[0]-10, posi[1]-10, 0])
    }
    sim.simulate(initial_conditions, 100)

    plt.show()