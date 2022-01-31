# Built-in imports
from __future__ import annotations
from dataclasses import dataclass
import os
import pickle
import time
from typing import Union

# Library imports
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Internal imports
from car.carModel import CarModel
from car.controller import Controller
from path.carVisualizer import CarVisualizer
from path.mapVisualizer import MapVisualizer
from path.trajectoryGenerator import TrajectoryGenerator
from sim_settings import SimSettings, def_car_constants, def_controller_parameters, TrajectoryPreset

# Dataclass for storing simulation data for posterior treatment
@dataclass
class SimData:
    settings: SimSettings
    trajectory: np.ndarray

    tracking_error_pos: np.ndarray
    tracking_error_vel: np.ndarray

    collisions: int

    simout: list[SimInstant]

# Dataclass for storing each simulation instant's important variables, for posterior treatment
@dataclass(frozen=True, order=True)
class SimInstant:
    time: float
    car_state: np.ndarray
    car_state_v_cm: np.ndarray
    sensors_output: np.ndarray
    controller_reference: np.ndarray
    controller_actuation: np.ndarray

    work_force: float
    energy_spent: float

class Simulator:
    ''' Simulator main class. 

    Handles the numerical simulation of the car model and envionment,
    aswell as instantiation and managment of all the important objects in the developed system's
    architecure.
    '''
    def __init__(self, settings: SimSettings):
        ''' Unpack the simulation settings data into the simulator's internal variables.

        Initialize all the objects with the supplied parameters.
        '''
        self.step_size_plot = settings.step_size_plot
        self.step_size_sim = settings.step_size_sim
        self.final_time_max = settings.sim_time
        self.seconds_to_film_after_end = settings.sim_time_after_stop

        self.car_model = CarModel(settings.car_constants)
        self.trajectory_generator = TrajectoryGenerator(
            settings.road_constants, settings.traj_endpoints, settings.energy_budget, self.car_model.M, self.car_model.idle_power)
        self.controller = Controller(settings.controller_parameters, self.car_model.L, self.trajectory_generator.energy_budget)
        self.energy_budget = self.trajectory_generator.energy_budget
        self.car_visualizer = CarVisualizer(settings.car_constants)
        self.map_visualizer = MapVisualizer(settings.road_constants)
        self.energy_spent = 0
        self.collisions = 0
        self.visualization = settings.visualization

        self.sim_time = settings.sim_time
        self.vis_window = settings.vis_window
        self.view_while_sim = settings.view_while_sim

        self.tracking_error_vel = []
        self.tracking_error_pos = []

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

        self.end_sim_triggered = False

    def update_data(self, time, car_state, car_state_v_cm, sensors_output,
                    controller_reference, controller_actuation, work_force, energy_spent):
        ''' Save current instant data into the simulation data structure.
        '''
        self.instants.append(SimInstant(
            time=time,
            car_state=car_state,
            car_state_v_cm=car_state_v_cm,
            sensors_output=sensors_output,
            controller_reference=controller_reference,
            controller_actuation=controller_actuation,
            work_force=work_force,
            energy_spent=energy_spent
        ))

        self.tracking_error_vel.append(self.trajectory_generator.states[controller_reference, 0] - car_state_v_cm[0])
        car_position = car_state[2:4]
        reference_trajectory = self.trajectory_generator.states[:, 2:4]
        if controller_reference == 0:
            tracking_error = np.linalg.norm(reference_trajectory[controller_reference] - car_position)
        else:
            # get line uniting to previous waypoint
            direction_vector = reference_trajectory[controller_reference, :] - reference_trajectory[controller_reference - 1, :]

            # if it has length 0 (the waypoints lie on top of eachother) just measure the distance to the waypoint
            if np.linalg.norm(direction_vector) == 0:
                tracking_error = np.linalg.norm(reference_trajectory[controller_reference] - car_position)
            # otherwise, measure the distance to the line
            else:
                # normalize direction vector
                direction_vector = direction_vector / np.linalg.norm(direction_vector)

                # get the car position relative to the previous waypoint
                relative_car_position = car_position - reference_trajectory[controller_reference - 1, :]

                # compute the distance along the trajectory of the relative car position
                distance_along_path = np.inner(relative_car_position, direction_vector)

                # compute error to trajectory using pythagoras
                tracking_error = np.sqrt(max(np.linalg.norm(relative_car_position)**2 - distance_along_path**2, 0))
        self.tracking_error_pos.append(tracking_error)

    def save_data(self, filename: str = 'sim_data.pkl', settings: Union[SimSettings, None] = None, video_file: str = 'simulation.mp4'):
        print('collisions', self.collisions)
        ''' Save the simulation's data to a cache file.
        '''
        sim_data = SimData(
            settings=settings,
            trajectory=np.array(self.trajectory_generator.states),
            tracking_error_pos=np.array(self.tracking_error_pos),
            tracking_error_vel=np.array(self.tracking_error_vel),
            simout=self.instants,
            collisions=self.collisions
        )
        with open(filename, 'wb') as f:
            pickle.dump(sim_data, f)
        if self.visualization:
            self.to_video(video_name=video_file)

    def to_file(self, iter: int):
        ''' Save the current frame into a file.
        '''
        try:
            plt.savefig(os.path.join(self.image_dir, f'{iter:04d}.png'))
        except:
            if os.path.isfile(os.path.join(self.image_dir, f'{iter:04d}.png')):
                os.remove(os.path.join(self.image_dir, f'{iter:04d}.png'))
            raise

    def to_video(self, video_name: str = 'simulation.mp4'):
        ''' Compile every frame into a video and save it to a file.
        '''
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
        ''' Run the entirety of the simulation.
        '''

        # Set the initial state of the dynamical system
        car_state = self.initial_conditions['car_ic']
        controller_output = np.array([0, 0])
        sensors_output = np.array([0, 0])
        trajectory_output = np.array([0, 0])

        # If the simulation is running in visualization mode, setup the window.
        if self.visualization:
            fig = plt.figure()
            ax: plt.Axes = fig.add_subplot(111)
            ax.set_position([0.02, 0.12, 0.8, 0.8])
            fig.canvas.draw()
            plt.show(block=False)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            self.map_visualizer.plot(ax)
            info_string = ""
            overlay = ax.text(0.74, 0.9, info_string, transform=fig.transFigure,
                              fontsize=10, verticalalignment='top', color='k')
            ti = time.time()

        # Simulate with a fixed time step and while the destination hasn't been achieved
        goal_achieved = False
        for instant in np.arange(self.sim_time, step=self.step_size_plot):
            t0 = time.time()
            for sim_instant in np.arange(instant, instant + self.step_size_plot, self.step_size_sim):

                # Accounts for mistmatches between the simulation step size and the plot step size
                if sim_instant > self.final_time_max:
                    return
                
                # When the simulation is finished, stop the dynamic simulation
                # but keep the visualization
                if goal_achieved:
                    if not self.end_sim_triggered:
                        self.final_time_max = sim_instant + self.seconds_to_film_after_end
                        self.end_sim_triggered = True
                    car_state[0] = 0
                    self.car_model.idle_power = 0
                else: 
                    car_input = controller_output

                    car_state = solve_ivp(self.car_model.derivative, (sim_instant, sim_instant + self.step_size_sim),
                                        car_state, args=(car_input,), method='RK45').y[:, -1]

                # Saturate phi to max turning angle
                if car_state[4] < -np.pi/3:
                    car_state[4] = -np.pi/3

                if car_state[4] > np.pi/3:
                    car_state[4] = np.pi/3

                car_output = self.car_model.output(sim_instant, car_state)

                trajectory_output = self.trajectory_generator.output(sim_instant)
                controller_input = [car_output, trajectory_output, self.energy_spent]
                controller_output, goal_achieved = self.controller.output(sim_instant, controller_input)

                controller_reference = self.controller.current_waypoint_idx

                # If the energy exceeds the energy budget, turn the car off
                if self.energy_spent >= self.energy_budget:
                    self.car_model.idle_power = 0

                # Calculate energy spent
                work_force = max(controller_output[0], 0)
                self.energy_spent += (work_force * car_state[0]
                                      + self.car_model.idle_power) * self.step_size_sim

                # Visualization
                self.car_visualizer.set_state(car_state)

                # Save data
                self.update_data(sim_instant, car_state, car_output, sensors_output,
                                 controller_reference, controller_output, work_force, self.energy_spent)


            # Determine collisions with environment
            self.collisions = self.map_visualizer.collision_counter(
                self.car_visualizer, visualization=self.visualization)

            # Due all sorts of visualizations things: plots, text info, etc.
            if not self.visualization:
                continue

            info_string = f'Time: {sim_instant:.2f} s\n'
            info_string += f'Energy spent/budget:\n'
            info_string += f'{self.energy_spent/1000:0.1f}/{self.energy_budget/1000:.1f} kJ\n'
            info_string += f'Velocity/Max Velocity:\n'
            info_string += f'{car_output[0]*3600/1000:.1f} / {trajectory_output[controller_reference][0]*3600/1000:.1f} km/h\n'
            info_string += f'Collisions: {self.collisions}\n'
            # Do some plots
            t1 = time.time()
            self.car_visualizer.plot(ax)
            self.controller.plot(ax)
            overlay.set_text(info_string)

            x, y = car_state[2:4]
            ax.set_xlim([self.vis_window[0][0] + x, self.vis_window[0][1] + x])
            ax.set_ylim([self.vis_window[1][0] + y, self.vis_window[1][1] + y])

            t2 = time.time()
            if self.view_while_sim:
                fig.canvas.flush_events()
                plt.show(block=False)

            t3 = time.time()
            self.to_file(int(instant/self.step_size_plot))
            t4 = time.time()
            print(f' {t1-t0:.2f} - {t2-t1:.2f} - {t3-t2:.2f} - {t4-t3:.2f} - total: {t4-t0:.2f}  {instant:.2f}/{self.sim_time:.2f} s ({(instant+self.step_size_plot)/self.sim_time*100:.2f}%) real time: {time.time() - ti:.2f}', end='\n')



if __name__ == "__main__":
    np.random.seed(1)

    # All the possible parameters are defined in sim_settings.py
    settings = SimSettings(
        # Visualization parameters
        step_size_plot=0.1,
        visualization=True,
        view_while_sim=True,

        # A relatively challenging trajectory
        traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
    )

    # Initiate the simulator object
    sim = Simulator(settings)

    # Run the simulation and if it's stopped save the video nonetheless
    try:
        sim.simulate()
    except KeyboardInterrupt:
        print('Simulation interrupted')
    finally:
        sim.save_data()
