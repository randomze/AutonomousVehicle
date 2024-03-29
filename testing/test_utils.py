from __future__ import annotations
import multiprocessing 
import os
import pickle
import sys

import numpy as np

from performance.cache_utils import cache_dir
from simulator import SimData, Simulator
from sim_settings import SimSettings, def_car_constants, def_controller_parameters

# Setup the default simulation folder
sims_folder = 'sims'
if not os.path.exists(os.path.join(cache_dir, sims_folder)):
    os.mkdir(os.path.join(cache_dir,sims_folder))

def run_sims(settings_list: list[SimSettings], batch_size: int = -1):
    """ Run a set of simulations in parallel, with as many simultaneous threads as the computer's
    CPU cores by default.
    """
    if batch_size == -1:
        batch_size = multiprocessing.cpu_count()

    args = []
    # For each simulation in the list, setup it's cache and prepare the arguments that must be
    # passed to the simulation runner in the proper way
    for settings in settings_list:
        key = str(hash(settings))
        sim_folder  = os.path.join(cache_dir, sims_folder, key )
        sim_data_file = os.path.join(sim_folder, 'sim_data.pkl')

        if not os.path.exists(sim_folder):
            os.mkdir(sim_folder)

        if os.path.exists(sim_data_file):
            continue

        sim_stdout_file = os.path.join(sim_folder, 'out.txt')

        args.append((settings, sim_data_file, sim_stdout_file))

    print(f"From {len(settings_list)} simulations, {len(settings_list) - len(args)} were already done")

    # Run the simulations concurrently with a pool of worker threads
    with multiprocessing.Pool(batch_size) as pool:
        pool.map(run_sim_star_wrapper, args)

def run_sim_star_wrapper(args):
    """ Wrapper function to circumvent the multiprocessing libraries interface.
    """
    return run_sim(*args)

def run_sim(settings: SimSettings, data_file: os.PathLike, stdout_file: os.PathLike = None):
    """ Run the simulation with the given settings, saving the output to the passed argument and
    eventually saving the output to the console to a file as well.
    """
    if not stdout_file is None:
        sys.stdout = open(stdout_file, 'w')

    sim = Simulator(settings)
    sim.simulate()
    sim.save_data(data_file, settings=settings)

def fetch_sim_data(settings: SimSettings) -> SimData:
    """ De-serialize simulation data from file into the Simulation Data data structures.
    """
    key = str(hash(settings))
    sim_folder  = os.path.join(cache_dir, sims_folder, key )
    sim_data_file = os.path.join(sim_folder, 'sim_data.pkl')
    if not os.path.exists(sim_data_file):
        print(f'Simulation {key} does not exist')
        return None
    with open(sim_data_file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    # A simple set of tests to validate the functioning of the functions of this module
    settings_lst = [
        SimSettings(), # default settings
        SimSettings(car_constants=def_car_constants(idle_power=10)),
        SimSettings(controller_parameters=def_controller_parameters(steering=120))
    ]


    settings_lst += [SimSettings(goal_crossing_distance=val) for val in np.arange(-3.5, -0.5, 0.1)]

    run_sims(settings_lst)

    sim_data = fetch_sim_data(settings_lst[0])

    sims_i_want = [SimSettings(), settings_lst[2]]

    sim_datas_i_want = [fetch_sim_data(s) for s in sims_i_want]

    steering_vals = [d.settings.controller_gains['steering'] for d in sim_datas_i_want]

    print(steering_vals)



