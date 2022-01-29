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
from sim_settings import SimSettings, def_car_constants, def_controller_gains, TrajectoryPreset


@dataclass
class SimData:
    settings: SimSettings
    trajectory: np.ndarray

    simout: list[SimInstant]


@dataclass(frozen=True, order=True)
class SimInstant:
    time: float
    car_state: np.ndarray
    car_state_v_cm: np.ndarray
    sensors_output: np.ndarray
    trajectory_output: np.ndarray
    controller_reference: np.ndarray

    work_force: float
    energy_spent: float
    collisions: int


class Simulator:
    def __init__(
            self, step_size_plot, step_size_sim, car_constants, map_constants, sensorParameters, controller_gains,
            path: tuple, time: float, energy_budget, goal_crossing_distance: float = -1,
            vis_window: tuple = ((-20, 20),
                                 (-20, 20)),
            visualization: bool = True, real_time=False):
        self.step_size_plot = step_size_plot
        self.step_size_sim = step_size_sim
        self.energy_budget = energy_budget

        self.car_model = CarModel(car_constants)
        self.controller = Controller(controller_gains, self.car_model.L, goal_crossing_distance=goal_crossing_distance)
        self.sensors = Sensors(sensorParameters)
        smoothen_window = 5
        self.trajectory_generator = TrajectoryGenerator(
            map_constants, path, smoothen_window, energy_budget, self.car_model.M, self.car_model.idle_power)
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
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)

        initial_waypoint_difference = self.trajectory_generator.path[1] - self.trajectory_generator.path[0]
        initial_heading = np.arctan2(initial_waypoint_difference[1], initial_waypoint_difference[0])
        self.initial_conditions = {
            'car_ic': np.array([0, initial_heading, *self.trajectory_generator.path[0], 0])
        }

        self.instants: list[SimInstant] = []

    def update_data(self, time, car_state, car_state_v_cm, sensors_output, trajectory_output,
                    controller_reference, work_force, energy_spent, collisions):
        self.instants.append(SimInstant(
            time=time,
            car_state=car_state,
            car_state_v_cm=car_state_v_cm,
            sensors_output=sensors_output,
            trajectory_output=trajectory_output,
            controller_reference=controller_reference,
            work_force=work_force,
            energy_spent=energy_spent,
            collisions=collisions
        ))

    def save_data(self, filename: str = 'sim_data.pkl', settings: Union[SimSettings, None] = None):
        sim_data = SimData(
            settings=settings,
            trajectory=self.trajectory_generator.path,
            simout=self.instants
        )
        with open(filename, 'wb') as f:
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
            overlay = ax.text(0.05, 0.95, info_string, transform=ax.transAxes,
                              fontsize=10, verticalalignment='top', color='y')
            ti = time.time()

        goal_achieved = False
        for instant in np.arange(self.sim_time, step=self.step_size_plot):
            t0 = time.time()
            for sim_instant in np.arange(instant, instant + self.step_size_plot, self.step_size_sim):
                if goal_achieved:
                    return
                car_input = controller_output

                car_state = solve_ivp(self.car_model.derivative, (sim_instant, sim_instant + self.step_size_sim),
                                      car_state, args=(car_input,), method='RK45').y[:, -1]

                # Saturate phi to max turning angle
                if car_state[4] < -np.pi/3:
                    car_state[4] = -np.pi/3

                if car_state[4] > np.pi/3:
                    car_state[4] = np.pi/3

                car_output = self.car_model.output(sim_instant, car_state)

                sensors_input = car_output
                sensors_output = self.sensors.output(sim_instant, sensors_input)

                trajectory_output = self.trajectory_generator.output(sim_instant)
                controller_input = [sensors_output, trajectory_output]
                controller_output, goal_achieved = self.controller.output(sim_instant, controller_input)

                controller_reference = self.controller.current_waypoint
                work_force = max(self.controller.force_apply, 0)
                self.energy_spent += (work_force * car_state[0]
                                      + self.car_model.idle_power) * self.step_size_sim
                self.car_visualizer.set_state(car_state)

                self.update_data(sim_instant, car_state, car_output, sensors_output, trajectory_output,
                                 controller_reference, work_force, self.energy_spent, self.collisions)

            self.collisions = self.map_visualizer.collision_counter(
                self.car_visualizer, visualization=self.visualization)

            if not self.visualization:
                continue

            info_string = f'Time: {sim_instant:.2f} s\n'
            info_string += f'Energy spent: {self.energy_spent:.2f} J\n'
            info_string += f'Energy budget: {self.energy_budget:.2f} J\n'
            info_string += f'Collisions: {self.collisions}\n'
            info_string += f'Velocity: {car_output[0]*3600/1000:.2f} / {controller_reference[0]*3600/1000:.2f} km/h\n'
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

class SimWrapperTests(Simulator):
    def __init__(self, settings: SimSettings):
        self.settings = settings

        super().__init__(settings.step_size_plot, settings.step_size_sim, settings.car_constants, settings.road_constants, settings.sensor_parameters, settings.controller_gains, settings.traj_endpoints, settings.sim_time, settings.energy_budget, settings.goal_crossing_distance, settings.vis_window, settings.visualization, settings.real_time)


if __name__ == "__main__":
    np.random.seed(1)

    settings = SimSettings(
        step_size_plot=0.2,
        step_size_sim=0.01,
        sim_time=100,

        traj_endpoints=TrajectoryPreset.VerySharpTurn.value,

        energy_budget=10e4,
        car_constants=def_car_constants(
            idle_power=0.1,
        ),
        controller_gains=def_controller_gains(
            deadzone_velocity_threshold=0.1,
            deadzone_continuity=True,
        ),

        visualization=True,
        real_time=True,
        vis_window=((-30, 30), (-30, 30)),
    )

    sim = SimWrapperTests(settings)

    try:
        sim.simulate()
    except KeyboardInterrupt:
        ...
    finally:
        sim.save_data()
