from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from sim_settings import SimSettings, TrajectoryPreset, def_controller_parameters
from tester import run_sims, fetch_sim_data
from controller_tests import tracking_error
from performance.cache_utils import cached


@dataclass
class SimBundle:
    sims = list[SimSettings]


def test_bundles(bundles: list[SimBundle]):
    sims = [simulation for bundle in bundles for simulation in bundle ]
    run_sims(sims)
    

def get_result(bundle: SimBundle):
    data = [fetch_sim_data(simulation) for simulation in bundle]
    if data is None:
        raise ValueError(f'Simulation data not found for {bundle}')
    return data

@cached(folder="bundle results")
def get_cost(bundle: SimBundle):
    data = get_result(bundle)

    mean_tracking_errors_pos = np.zeros(len(data))
    mean_tracking_errors_vel = np.zeros(len(data))
    mean_abs_tracking_errors_vel = np.zeros(len(data))

    car_state_data = [[instant.car_state_v_cm for instant in data.simout] for data in data]
    reference_data = [[instant.controller_reference for instant in data.simout] for data in data]
    trajectory_data = [data.trajectory for data in data]

    for simulation in range(len(data)):
        car_state = np.vstack(car_state_data[simulation])
        reference = np.vstack(reference_data[simulation])
        trajectory = trajectory_data[simulation]

        car_trajectory = car_state[:, 2:4]
        car_velocities = car_state[:, 0].reshape((car_state.shape[0], 1))
        reference_velocities = trajectory[reference, 0]

        tracking_error_pos = tracking_error(car_trajectory, trajectory, reference)
        tracking_error_vel = reference_velocities - car_velocities

        mean_tracking_errors_pos[simulation] = np.mean(tracking_error_pos)
        mean_tracking_errors_vel[simulation] = np.mean(tracking_error_vel)
        mean_abs_tracking_errors_vel[simulation] = np.mean(np.abs(tracking_error_vel))

    return np.mean(mean_tracking_errors_pos), np.mean(mean_tracking_errors_vel), np.mean(mean_abs_tracking_errors_vel)

def make_trajectories_bundle(settings: SimSettings):
    bundle_settings = []
    trajectories = [preset.value for preset in TrajectoryPreset]

    for trajectory in trajectories:
        settings_copy = copy.copy(settings)
        settings_copy.traj_endpoints = trajectory
        bundle_settings.append(settings_copy)

    return bundle_settings

if __name__ == '__main__':
    parameter_vars = [
        SimSettings(controller_parameters=def_controller_parameters(steering=force_vals))
        for force_vals in np.logspace(-2, 2, num=5)
    ]
    bundles = [make_trajectories_bundle(settings) for settings in parameter_vars]


    test_bundles(bundles)
    costs = [get_cost(bundle) for bundle in bundles]

    # get each cost for each force_val and print them together
    for idx, force_val in enumerate(np.logspace(-2, 2, num=5)):
        print(f'{force_val} {costs[idx]}')



