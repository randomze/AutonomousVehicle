from __future__ import annotations

import copy
from dataclasses import dataclass
import multiprocessing
from typing import Collection

import numpy as np

from performance.cache_utils import cached
from simulator import SimData
from sim_settings import SimSettings, TrajectoryPreset, def_controller_parameters
from testing.test_utils import run_sims, fetch_sim_data

# A wrapper class that groups sets of simulations in bundles. Useful for testing the same
# controller on various different trajectories and keeping that data bundled, for comparison
# with other controllers.
@dataclass
class SimBundle:
    sims = list[SimSettings]


def test_bundles(bundles: list[SimBundle]):
    """ Unpack the various bundles into a single list of tests to be ran.
    (The usefulness of the bundles isn't lost by unpacking since the simulation data
    which is later used is fetched using the bundle's hash, preserving the bundling.)
    """
    sims = []
    for bundle in bundles:
        for sim in bundle:
            sims.append(sim)
    run_sims(sims)
    

def get_result(bundle: SimBundle):
    """ Fetch a bundle's result from the saved data files.
    """
    data = [fetch_sim_data(simulation) for simulation in bundle]
    if data is None:
        raise ValueError(f'Simulation data not found for {bundle}')
    return data

def queue_friendly_cost(bundle):
    """ Wrapper function for the cost calculations for multi-processing.
    """
    return get_cost(bundle)

def get_cost_components(bundles: list[SimBundle]):
    """ Get the cost function's arguments for a list of bundles.

    Do it concurrently to speed up the process.
    """
    batch_size = multiprocessing.cpu_count() - 1

    with multiprocessing.Pool(batch_size) as pool:
        pool.map(queue_friendly_cost, bundles, chunksize=10)

    return [get_cost(bundle) for bundle in bundles]

@cached(folder="bundle results")
def get_cost(bundle: SimBundle):
    """ Calculate each argument of a bundle's cost:

    Average position error, across each simulation and across the bundle.
    Average velocity error, across each simulation and across the bundle.
    Average absolute velocity error, across each simulation and across the bundle.
    Maximum power of the whole bundle.
    Maximum actuation force of the whole bundle.
    Maximum actuation steering of the whole bundle.
    """
    data: list[SimData] = get_result(bundle)

    for sim in data:
        if sim.collisions != 0:
            return np.inf

    errors_pos = []
    errors_vel = []
    errors_vel_abs = []
    max_actuation_steering = []
    max_actuation_force = []
    max_power = []
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
        max_power.append(np.max([
            np.abs(max(instant.car_state[0]*instant.controller_actuation[0], 0))
            for instant in sim.simout]
        ))

    errors_pos = np.mean(errors_pos)
    errors_vel = np.mean(errors_vel)
    errors_vel_abs = np.mean(errors_vel_abs)
    max_actuation_force = np.max(max_actuation_force)
    max_actuation_steering = np.max(max_actuation_steering)
    max_power = np.max(max_power)

    return errors_pos, errors_vel, errors_vel_abs, max_power, max_actuation_force, max_actuation_steering

def cost_fcn(args, gains: Collection):
    """ Compute the cost of a bundle given the components and the set of gains which
    characterize the cost function.
    """
    if not isinstance(args, Collection):
        if args == np.inf:
            return np.inf
        else:
            raise ValueError(f'args must be a Collection of SimBundle or infinity, not {args}')

    gains = np.array(gains)

    cost = np.dot(gains, np.array(args))

    return cost

def make_trajectories_bundle(settings: SimSettings):
    """ Generate a bundle of simulations with the same settings but with different trajectories.
    """
    bundle_settings = []
    trajectories = [preset.value for preset in TrajectoryPreset]

    for trajectory in trajectories:
        settings_copy = copy.copy(settings)
        settings_copy.traj_endpoints = trajectory
        bundle_settings.append(settings_copy)

    return bundle_settings

def show_bundle_results(bundles: list[SimBundle], cost_components: Collection, cost: np.ndarray, idxs: np.ndarray, gains: np.ndarray):
    """ Show the cost of each bundle, as well as it's components and the associated relevant controller parameters.
    """
    components_str = []
    for components_bundle in cost_components:
        if not isinstance(components_bundle, np.ndarray) and components_bundle == np.inf:
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

def gains_to_normalize(cost_components: np.ndarray):
    """ Determines the normalization factor for each cost component, so that no single cost component,
    across all bundles and simulations, can be greater than 1.
    """

    gains = np.zeros(cost_components.shape[1])
    for i in range(cost_components.shape[1]):
        max_val = np.max(np.abs(cost_components[:,i]))
        if max_val == 0:
            gains[i] = 1
        else:
            gains[i] = 1/max_val

    return gains

if __name__ == '__main__':
    parameter_vars = [
        SimSettings(controller_parameters=def_controller_parameters(
            steering=steering_vals,
            force=733.33,
            goal_crossing_distance=-2.0,
        ))
        for goal_crossing_distance_vals in np.linspace(-3, 0, num=10)
        for force_vals in np.linspace(100, 2000, num=25)
        for steering_vals in np.linspace(1, 200, num=15)
    ]

    # Costs chosen for the cost function to choose the best controller and definition of the maximum
    # acceptable front wheel torque
    cost_fcn_gains = np.array((1, 1/2, 1/2, 1, 1, 1))
    car_max_torque = 5000
    wheel_radius = 0.256

    # Generate simulation settings
    bundles = [make_trajectories_bundle(settings) for settings in parameter_vars]

    # Perform all simulations, or make sure cached results exist
    test_bundles(bundles)

    # Get cost information for the simulations and print results
    cost_components_raw = get_cost_components(bundles)

    print("cost components raw: ", len(cost_components_raw))

    cost_components = []
    good_bundles = []
    for idx, component in enumerate(cost_components_raw):
        if component == np.inf:
            continue
        print(component)
        good_bundles.append(bundles[idx])
        cost_components.append(component)
    
    print("cost components: ", len(cost_components))
    cost_components_filtered = []
    actually_good_bundles = []
    for idx, component in enumerate(cost_components):
        if component[4]*wheel_radius < car_max_torque: # ignore cases where there's unrealistic torque
            cost_components_filtered.append(component)
            actually_good_bundles.append(good_bundles[idx])

    
    print("cost components filtered: ", len(cost_components_filtered))
    cost_components = np.abs(np.array(cost_components_filtered))
    print("cost components: ", len(cost_components))
    norm_gains = gains_to_normalize(cost_components)

    cost_components = cost_components * norm_gains

    maxes = np.array([1/norm_gain for norm_gain in norm_gains])
    print(f"{maxes =}")

    print(f'Invalid bundles (in which car collided or max torque achieved): {len(cost_components_raw) - len(cost_components)}/{len(cost_components_raw)}')

    costs = np.array([cost_fcn(component, cost_fcn_gains) for component in cost_components])

    idxs = np.argsort(costs).tolist()

    #select only first 10 and last 10 indexes
    idxs_best = np.array(idxs[:10])
    idx_worst = np.array(idxs[-10:])

    print('Best controller parameters:')
    show_bundle_results(actually_good_bundles, cost_components, costs, idxs_best, cost_fcn_gains)

    print('Worst controller parameters:')
    show_bundle_results(actually_good_bundles, cost_components, costs, idx_worst, cost_fcn_gains)





