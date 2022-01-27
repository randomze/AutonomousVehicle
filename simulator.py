from __future__ import annotations
import os
import pickle
import time
from dataclasses import dataclass
from typing import Union
import imageio
from model.carModel import CarModel
from model.controller import Controller
from model.sensors import Sensors
from model.trajectoryGenerator import TrajectoryGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from visualization.carVisualizer import CarVisualizer
from visualization.mapVisualizer import MapVisualizer
from model.physics import MoI, CoM_position
from sim_settings import SimSettings

@dataclass
class SimData:
    settings : SimSettings
    trajectory: np.ndarray

    simout: list[SimInstant]


@dataclass(frozen=True, order=True)
class SimInstant:
    time: float
    car_state: np.ndarray
    car_state_v_cm: np.ndarray
    sensors_output: np.ndarray

    work_force: float
    energy_spent: float
    collisions: int


class Simulator:
    def __init__(self, step_size_plot, step_size_sim, car_constants, map_constants, sensorParameters, controller_gains, path: tuple, time: float, energy_budget, goal_crossing_distance: float = -1, vis_window: tuple = ((-20, 20), (-20, 20)), visualization: bool = True, real_time = False):
        self.step_size_plot = step_size_plot
        self.step_size_sim = step_size_sim
        self.energy_budget = energy_budget

        self.car_model = CarModel(car_constants)
        self.controller = Controller(controller_gains, goal_crossing_distance=goal_crossing_distance)
        self.sensors = Sensors(sensorParameters)
        smoothen_window = 5
        self.trajectory_generator = TrajectoryGenerator(map_constants, path, time, smoothen_window, energy_budget)
        self.car_visualizer = CarVisualizer(car_constants)
        self.map_visualizer = MapVisualizer(map_constants)
        self.energy_spent = 0
        self.collisions = 0
        self.visualization = visualization

        self.sim_time = time
        self.vis_window = vis_window
        self.realtime = real_time

        self.cache_dir = os.path.join('cache')
        self.image_dir = os.path.join(self.cache_dir, 'images')
        if not os.path.isdir(self.cache_dir): os.mkdir(self.cache_dir)
        if not os.path.isdir(self.image_dir): os.mkdir(self.image_dir)

        initial_waypoint_difference = self.trajectory_generator.path[1] - self.trajectory_generator.path[0]
        initial_heading = np.arctan2(initial_waypoint_difference[1], initial_waypoint_difference[0])
        self.initial_conditions = {
            'car_ic': np.array([0, initial_heading, *self.trajectory_generator.path[0], 0])
        }


        self.instants: list[SimInstant] = []


    def update_data(self, time, car_state, car_state_v_cm, sensors_output, work_force, energy_spent, collisions):
        self.instants.append(SimInstant(
            time            =time,
            car_state       =car_state,
            car_state_v_cm  =car_state_v_cm,
            sensors_output  =sensors_output,
            work_force      =work_force,
            energy_spent    =energy_spent,
            collisions      =collisions
        ))

    def save_data(self, filename: str = 'sim_data.pkl', settings: Union[SimSettings, None] = None):
        sim_data = SimData(
            settings = settings,
            trajectory = self.trajectory_generator.path,
            simout = self.instants
        )
        with open(os.path.join(self.cache_dir, filename), 'wb') as f:
            pickle.dump(sim_data, f)
        if self.visualization:
            self.to_video()

    def to_file(self, iter: int):
        try:
            plt.savefig(os.path.join(self.image_dir, f'{iter:04d}.png'))
        except:
            if os.path.isfile(os.path.join(self.image_dir, f'{iter:04d}.png')):
                os.remove(os.path.join(self.image_dir, f'{iter:04d}.png'))
            raise
    
    def to_video(self, video_name: str = 'simulation.mp4'):
        print('Saving video...')
        fps = int(1/self.step_size_plot)
        images = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                images.append(os.path.join(self.image_dir, filename))
        images.sort()
        writer = imageio.get_writer(video_name, fps=fps)
        for image in images:
            writer.append_data(imageio.imread(image))
            os.remove(image)
        writer.close()

    def simulate(self):
        car_state = self.initial_conditions['car_ic']
        controller_output = np.array([0, 0])
        sensors_output = np.array([0, 0])
        trajectory_output = np.array([0, 0])
        
        if self.visualization:
            fig = plt.figure()
            ax: plt.Axes = fig.add_subplot(111)
            fig.canvas.draw()
            plt.show(block=False)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            self.map_visualizer.plot(ax)
            info_string = ""
            overlay = ax.text(0.05, 0.95, info_string, transform=ax.transAxes, fontsize=10, verticalalignment='top', color='y')
            ti = time.time()

        for instant in np.arange(self.sim_time, step=self.step_size_plot):
            t0 = time.time()
            for sim_instant in np.arange(instant, instant + self.step_size_plot, self.step_size_sim):

                car_input = controller_output
                
                car_state = solve_ivp(self.car_model.derivative, (sim_instant, sim_instant + self.step_size_sim), car_state, args=(car_input,), method='RK45').y[:,-1]
                
                # Saturate phi in order not to reach weird stuff
                if car_state[4] < -np.pi/3:
                    car_state[4] = -np.pi/3

                if car_state[4] > np.pi/3:
                    car_state[4] = np.pi/3

                car_output = self.car_model.output(sim_instant, car_state)

                sensors_input = car_output
                sensors_output = self.sensors.output(sim_instant, sensors_input)

                trajectory_output = self.trajectory_generator.output(sim_instant)
                controller_input = [sensors_output, trajectory_output]
                controller_output = self.controller.output(sim_instant, controller_input)

                work_force = self.controller.force_apply if self.controller.force_apply > 0 else 0
                self.energy_spent += (work_force * car_output[0] 
                                    + self.car_model.idle_power) * self.step_size_sim
                self.car_visualizer.set_state(car_state)

                self.update_data(sim_instant, car_state, car_output, sensors_output, work_force, self.energy_spent, self.collisions)
            
            self.collisions = self.map_visualizer.collision_counter(self.car_visualizer)


            if not self.visualization: continue

            info_string = f'Time: {sim_instant:.2f} s\n'
            info_string += f'Energy spent: {self.energy_spent:.2f} J\n'
            info_string += f'Energy budget: {self.energy_budget:.2f} J\n'
            info_string += f'Collisions: {self.collisions}\n'
            info_string += f'Velocity: {car_output[0]*3600/1000:.2f} km/h\n'
            # Do some plots
            t1 = time.time()
            self.car_visualizer.plot(ax)
            self.controller.plot(ax)
            overlay.set_text(info_string)

            x, y = car_state[2:4]
            ax.set_xlim([self.vis_window[0][0] + x, self.vis_window[0][1] + x])
            ax.set_ylim([self.vis_window[1][0] + y, self.vis_window[1][1] + y])

            t2 = time.time()
            if self.realtime:
                fig.canvas.flush_events()
                plt.show(block=False)

            t3 = time.time()
            self.to_file(int(instant/self.step_size_plot))
            t4 = time.time()
            print(f' {t1-t0:.2f} - {t2-t1:.2f} - {t3-t2:.2f} - {t4-t3:.2f} - total: {t4-t0:.2f}  {instant:.2f}/{self.sim_time:.2f} s ({(instant+self.step_size_plot)/self.sim_time*100:.2f}%) real time: {time.time() - ti:.2f}', end='\n')

if __name__ == "__main__":
    road_constants = {
        'lat': 38.7367256,
        'lon': -9.1388871,
        'zoom': 16,
        'upsampling': 3,
        'regularization': 5,
    }
    
    posi = (-5, 10)
    posf = (-85, 100)
    sim_time = 100
    energy_budget = 10000
    plot_step = 0.4
    sim_step = 0.01

    view_sim_realtime = True # setting to false halves visualization overhead

    goal_crossing_distance = -2.54

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
        'M': 810.0,
        'Izz': Izz,
        'r_cm': com_r,
        'delta_cm': com_delta,
        'wheel_width': 0.1,
        'idle_power' : 1
    }  
    controller_gains = {
        'force': 1000,
        'steering': 100,
    }
    sim = Simulator(plot_step, sim_step, car_constants, road_constants, None, controller_gains, (posi, posf), sim_time, energy_budget, goal_crossing_distance=goal_crossing_distance, vis_window=((-20, 20), (-20, 20)), real_time=view_sim_realtime)
    
    try:
        sim.simulate()
    except KeyboardInterrupt:
        ...
    finally:
        sim.save_data()
