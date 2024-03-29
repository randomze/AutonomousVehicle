from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sim_settings import SimSettings, def_car_constants, def_controller_parameters
from testing.test_utils import run_sims, fetch_sim_data

def tracking_error(car_trajectory: np.ndarray, trajectory_output: np.ndarray, controller_reference: np.ndarray):
    """ Compute the tracking error to the path, disregarding the time reference.
    """
    tracking_error = np.empty(car_trajectory.shape[0])
    reference_trajectory = trajectory_output[:, 2:4]

    # For each reference, determine the segment the car is in and measure the distance
    # to the path.
    for index in range(len(controller_reference)):
        car_position = car_trajectory[index]
        current_reference = controller_reference[index]

        if current_reference == 0:
            tracking_error[index] = np.linalg.norm(reference_trajectory[current_reference] - car_position)
        else:
            # Get line uniting to previous waypoint
            direction_vector = reference_trajectory[current_reference, :] - reference_trajectory[current_reference - 1, :]

            # If it has length 0 (the waypoints lie on top of each other) just measure the distance to the waypoint
            if np.linalg.norm(direction_vector) == 0:
                tracking_error[index] = np.linalg.norm(reference_trajectory[current_reference] - car_position)
            # Otherwise, measure the distance to the line
            else:
                # Normalize direction vector
                direction_vector = direction_vector / np.linalg.norm(direction_vector)

                # Get the car position relative to the previous waypoint
                relative_car_position = car_position - reference_trajectory[current_reference - 1, :]

                # Compute the distance along the trajectory of the relative car position
                distance_along_path = np.inner(relative_car_position, direction_vector)

                # Compute error to trajectory using Pythagoras
                tracking_error[index] = np.sqrt(max(np.linalg.norm(relative_car_position)**2 - distance_along_path**2, 0))


    return tracking_error

def time_to_distance(time: np.ndarray, car_velocities: np.ndarray):
    """ Convert simulation time into distance traveled using the car's velocity.
    """
    time_step = time[1] - time[0]

    distance = np.empty_like(time)
    distance[0] = 0

    for index in range(time.shape[0] - 1):
        distance[index + 1] = distance[index] + car_velocities[index] * time_step

    return distance

if __name__=='__main__':
    # A set of controller parameters for which to plot things
    settings_list = [
        SimSettings(controller_parameters=def_controller_parameters(force=10)),
        SimSettings(controller_parameters=def_controller_parameters(force=100)),
        SimSettings(controller_parameters=def_controller_parameters(force=1000)),
        SimSettings(controller_parameters=def_controller_parameters(steering=1)),
        SimSettings(controller_parameters=def_controller_parameters(steering=10)),
        SimSettings(controller_parameters=def_controller_parameters(steering=100))
    ]

    run_sims(settings_list)

    # Unpack simulation data into easier variables to work with
    simulation_data = [fetch_sim_data(s) for s in settings_list]

    car_state_data = [[instant.car_state_v_cm for instant in data.simout] for data in simulation_data]
    reference_data = [[instant.controller_reference for instant in data.simout] for data in simulation_data]
    time_data = [[instant.time for instant in data.simout] for data in simulation_data]
    trajectory_data = [data.trajectory for data in simulation_data]

    controller_gains = [(data.settings.controller_parameters['force'], data.settings.controller_parameters['steering']) for data in simulation_data]

    # For each controller tested plot variables of interest
    for simulation in range(len(simulation_data)):
        car_state = np.vstack(car_state_data[simulation])
        reference = np.vstack(reference_data[simulation])
        trajectory = trajectory_data[simulation]
        time = np.array(time_data[simulation])

        car_velocities = car_state[:, 0].reshape((car_state.shape[0], 1))
        reference_velocities = trajectory[reference, 0]

        velocity_error = reference_velocities - car_velocities
        force_gain, steering_gain = controller_gains[simulation]
        plt.figure(0)
        plt.plot(time, velocity_error, label=f"K_f={force_gain}, K_s={steering_gain}")
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Error [m s^-1]')

        car_trajectory = car_state[:, 2:4]
        tracking_error_data = tracking_error(car_trajectory, trajectory, reference)
        distance = time_to_distance(time, car_state[:, 0])

        plt.figure(1)
        plt.plot(distance, tracking_error_data, label=f"K_f={force_gain}, K_s={steering_gain}")
        plt.xlabel('Distance [m]')
        plt.ylabel('Tracking Error [m]')

    plt.legend(loc="upper right")
    plt.show()