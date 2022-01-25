import os
import time
import copy
import threading
from typing import Tuple
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

class Simulator:
    def __init__(self, step_size_plot, step_size_sim, car_constants, map_constants, sensorParameters, path: tuple, time: float, energy_budget, goal_crossing_distance: float = -1):
        self.step_size_plot = step_size_plot
        self.step_size_sim = step_size_sim
        self.energy_budget = energy_budget

        self.car_model = CarModel(car_constants)
        self.controller = Controller(goal_crossing_distance=goal_crossing_distance)
        self.sensors = Sensors(sensorParameters)
        smoothen_window = 5
        self.trajectory_generator = TrajectoryGenerator(map_constants, path, time, smoothen_window, energy_budget)
        self.car_visualizer = CarVisualizer(car_constants)
        self.map_visualizer = MapVisualizer(map_constants)
        self.energy_spent = 0

        self.cache_dir = os.path.join('cache')
        self.image_dir = os.path.join(self.cache_dir, 'images')
        if not os.path.isdir(self.cache_dir): os.mkdir(self.cache_dir)
        if not os.path.isdir(self.image_dir): os.mkdir(self.image_dir)

        self.img_saving_threads = []

    def to_file(self, iter: int, threaded: bool = False):
        if threaded: # doesn't work. variation might?
            # get current figure, copy it, and save the figure using separate thread
            fig = copy.deepcopy(plt.gcf())
            
            def func(figure, path):
                figure.canvas.flush_events()
                figure.savefig(path)

            args = (fig, os.path.join(self.image_dir, f'{iter:04d}.png'))
            thread = threading.Thread(target=func, args=args)
            thread.start()
            self.img_saving_threads.append(thread)
        else:
            try:
                plt.savefig(os.path.join(self.image_dir, f'{iter:04d}.png'))
            except:
                if os.path.isfile(os.path.join(self.image_dir, f'{iter:04d}.png')):
                    os.remove(os.path.join(self.image_dir, f'{iter:04d}.png'))
                raise
    
    def to_video(self, fps: int, video_name: str = 'simulation.mp4'):
        print('Saving video...')
        for thread in self.img_saving_threads:
            thread.join()
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

    def simulate(self, initial_conditions, final_time, vis_window = ((-20, 20), (-20, 20)), real_time = False):
        car_state = initial_conditions['car_ic']
        controller_output = np.array([0, 0])
        sensors_output = np.array([0, 0])
        trajectory_output = np.array([0, 0])
        
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
        try:
            for instant in np.arange(final_time, step=self.step_size_plot):
                t0 = time.time()
                for sim_instant in np.arange(instant, instant + self.step_size_plot, self.step_size_sim):

                    car_input = controller_output
                    
                    car_state = solve_ivp(self.car_model.derivative, (sim_instant, sim_instant + self.step_size_sim), car_state, args=(car_input,), method='RK45').y[:,-1]
                    
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


                info_string = f'Time: {sim_instant:.2f} s\n'
                info_string += f'Energy spent: {self.energy_spent:.2f} J\n'
                info_string += f'Energy budget: {self.energy_budget:.2f} J\n'

                overlay.set_text(info_string)

                # Do some plots
                t1 = time.time()
                self.car_visualizer.plot(ax)
                self.controller.plot(ax)

                x, y = car_state[2:4]
                ax.set_xlim([vis_window[0][0] + x, vis_window[0][1] + x])
                ax.set_ylim([vis_window[1][0] + y, vis_window[1][1] + y])

                t2 = time.time()
                if real_time:
                    fig.canvas.flush_events()
                    plt.show(block=False)

                t3 = time.time()
                self.to_file(int(instant/self.step_size_plot))
                t4 = time.time()
                print(f' {t1-t0:.3f} - {t2-t1:.2f} - {t3-t2:.2f} - {t4-t3:.2f} - total: {t4-t0:.2f}  {instant:.2f}/{final_time:.2f} s ({(instant+self.step_size_plot)/final_time*100:.2f}%) real time: {time.time() - ti:.2f}', end='\n')
        except:
            self.to_video(fps=int(1/self.step_size_plot))
            raise
        self.to_video(fps=int(1/self.step_size_plot))

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
    energy_budget = 1000
    plot_step = 0.3
    sim_step = 0.001

    view_sim_realtime = True # setting to false halves execution time. images can be seen in folder

    goal_crossing_distance = -3.2

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
        'wheel_width': 0.1,
        'idle_power' : 1
    }  
    sim = Simulator(plot_step, sim_step, car_constants, road_constants, None, (posi, posf), sim_time, energy_budget, goal_crossing_distance=goal_crossing_distance)

    initial_conditions = {
        'car_ic': np.array([0, 0, posi[0]-10, posi[1]-10, 0])
    }
    sim.simulate(initial_conditions, sim_time, vis_window=((-20, 20), (-20, 20)), real_time=view_sim_realtime)
