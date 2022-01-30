from __future__ import annotations

import copy
from dataclasses import dataclass
import multiprocessing
from typing import Collection

import numpy as np

from sim_settings import SimSettings, TrajectoryPreset, def_controller_parameters
from simulator import SimData
from tester import run_sims, fetch_sim_data
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

def queue_friendly_cost(bundle):
    return get_cost(bundle)

def get_cost_components(bundles: list[SimBundle]):
    batch_size = multiprocessing.cpu_count() - 1

    with multiprocessing.Pool(batch_size) as pool:
        pool.map(queue_friendly_cost, bundles, chunksize=10)

    return [get_cost(bundle) for bundle in bundles]

@cached(folder="bundle results")
def get_cost(bundle: SimBundle):
    data: list[SimData] = get_result(bundle)

    completion_time = np.mean([sim.simout[-1].time/sim.settings.sim_time for sim in data])

    for sim in data:
        if sim.collisions != 0:
            return np.inf

    errors_pos = []
    errors_vel = []
    errors_vel_abs = []
    max_actuation_steering = []
    max_actuation_force = []
    for sim in data:
        errors_pos.append(np.mean(sim.tracking_error_pos))
        errors_vel.append(np.mean(sim.tracking_error_vel))
        errors_vel_abs.append(np.mean(np.abs(sim.tracking_error_vel)))
        max_actuation_force.append(np.max([
            np.abs(instant.controller_actuation[0]) 
            for instant in sim.simout]
        ))
        max_actuation_steering.append(np.max([
            np.abs(instant.controller_actuation[1]) 
            for instant in sim.simout]
        ))

    errors_pos = np.mean(errors_pos)
    errors_vel = np.mean(errors_vel)
    errors_vel_abs = np.mean(errors_vel_abs)
    max_actuation_force = np.max(max_actuation_force)
    max_actuation_steering = np.max(max_actuation_steering)


    return errors_pos, errors_vel, errors_vel_abs, completion_time, max_actuation_force, max_actuation_steering

def cost_fcn(args, gains: Collection):
    if not isinstance(args, Collection):
        if args == np.inf:
            return np.inf
        else:
            raise ValueError(f'args must be a Collection of SimBundle or infinity, not {args}')

    gains = np.array(gains)

    cost = np.dot(gains, np.array(args))

    return cost

def make_trajectories_bundle(settings: SimSettings):
    bundle_settings = []
    trajectories = [preset.value for preset in TrajectoryPreset]

    for trajectory in trajectories:
        settings_copy = copy.copy(settings)
        settings_copy.traj_endpoints = trajectory
        bundle_settings.append(settings_copy)

    return bundle_settings

def show_bundle_results(bundles: list[SimBundle], cost_components: Collection, cost: np.ndarray, idxs: np.ndarray, gains: np.ndarray):
    components_str = []
    for components_bundle in cost_components:
        if components_bundle == np.inf:
            components_str.append('inf')
        else:
            components_times_gain = [f'{components_bundle[i]*gains[i]:.2f}' for i in range(len(components_bundle))]
            components_raw = [f'{components_bundle[i]:.2f}' for i in range(len(components_bundle))]
            components_str.append([f'{components_times_gain[i]}/{components_raw[i]}' for i in range(len(components_bundle))])

    params_show_best = [(f'cost: {cost[idx]:.2f}',
                        f'components = {components_str[idx]}',
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

    cost_fcn_gains = np.array((1, 1, 1, 0, 1e-6, 1/20))

    bundles = [make_trajectories_bundle(settings) for settings in parameter_vars]

    test_bundles(bundles) # perform all simulations, or make sure cached results exist

    cost_components_raw = get_cost_components(bundles)

    cost_components = []
    for component in cost_components_raw:
        if component == np.inf:
            continue
        cost_components.append(component)

    print(f'Invalid bundles (in which car collided): {len(cost_components_raw) - len(cost_components)}/{len(cost_components_raw)}')

    costs = np.array([cost_fcn(component) for component in cost_components])

    idxs = np.argsort(costs).tolist()

    #select only first 10 and last 10 indexes
    idxs_best = np.array(idxs[:10])
    idx_worst = np.array(idxs[-10:])

    print('Best controller parameters:')
    show_bundle_results(bundles, cost_components, costs, idxs_best, cost_fcn_gains)

    print('Worst controller parameters:')
    show_bundle_results(bundles, cost_components, costs, idx_worst, cost_fcn_gains)





