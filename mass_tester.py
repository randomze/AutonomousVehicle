from __future__ import annotations

import copy
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np

from sim_settings import SimSettings, TrajectoryPreset, def_controller_parameters
from simulator import SimData
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
    data: list[SimData] = get_result(bundle)

    for sim in data:
        if sim.collisions != 0:
            return (np.inf, np.inf, np.inf)

    errors_pos = []
    errors_vel = []
    errors_vel_abs = []
    for sim in data:
        errors_pos.append(np.mean(sim.tracking_error_pos))
        errors_vel.append(np.mean(sim.tracking_error_vel))
        errors_vel_abs.append(np.mean(np.abs(sim.tracking_error_vel)))

    errors_pos = np.mean(errors_pos)
    errors_vel = np.mean(errors_vel)
    errors_vel_abs = np.mean(errors_vel_abs)

    return errors_pos, errors_vel, errors_vel_abs

def cost_fcn(mean_error_pos, mean_error_vel, mean_error_vel_abs):
    return mean_error_pos + mean_error_vel + mean_error_vel_abs

def make_trajectories_bundle(settings: SimSettings):
    bundle_settings = []
    trajectories = [preset.value for preset in TrajectoryPreset]

    for trajectory in trajectories:
        settings_copy = copy.copy(settings)
        settings_copy.traj_endpoints = trajectory
        bundle_settings.append(settings_copy)

    return bundle_settings

def show_bundle_results(bundles: list[SimBundle], cost: np.ndarray, idxs: np.ndarray):
    params_show_best = [(f'cost: {cost[idx]:.2f}',
                        str('steering: ') + str(bundles[idx][0].controller_parameters['steering']),
                        str('force: ') + str(bundles[idx][0].controller_parameters['force']),
                        str('goal crossing: ') + str(bundles[idx][0].controller_parameters['goal_crossing_distance']))
                        for idx in idxs
    ]

    print('\n'.join([str(param) for param in params_show_best])) 
    

if __name__ == '__main__':
    parameter_vars = [
        SimSettings(controller_parameters=def_controller_parameters(
            steering=steering_vals,
            force=force_vals,
            goal_crossing_distance=goal_crossing_distance_vals,
        ))
        for goal_crossing_distance_vals in np.linspace(-3, -1, num=5)
        for force_vals in np.linspace(100, 2000, num=25)
        for steering_vals in np.linspace(10, 200, num=10)
    ]
    parameter_vars += [
        SimSettings(controller_parameters=def_controller_parameters(
            steering=steering_vals,
            force=force_vals,
            goal_crossing_distance=goal_crossing_distance_vals,
        ))
        for goal_crossing_distance_vals in np.linspace(-1, 0, num=5)
        for force_vals in np.linspace(2000, 20000, num=25)
        for steering_vals in np.linspace(1, 10, num=10)
    ]


    bundles = [make_trajectories_bundle(settings) for settings in parameter_vars]

    test_bundles(bundles)
    costs = np.array([cost_fcn(*get_cost(bundle)) for bundle in bundles])

    print(costs.shape)

    idxs = np.argsort(costs).tolist()

    #select only first 10 and last 10 indexes
    idxs_best = np.array(idxs[:10])
    idx_worst = np.array(idxs[-10:])

    print('Best controller parameters:')
    show_bundle_results(bundles, costs, idxs_best)

    print('Worst controller parameters:')
    show_bundle_results(bundles, costs, idx_worst)





