from __future__ import annotations
import dataclasses
import json
import sys
import os
import pickle
import multiprocessing 
import time

import numpy as np
from performance.cache_utils import cache_dir
from simulator import SimInstant, SimData, SimWrapperTests
from sim_settings import SimSettings, def_car_constants, def_controller_parameters

sims_folder = 'sims'
if not os.path.exists(os.path.join(cache_dir, sims_folder)):
    os.mkdir(os.path.join(cache_dir,sims_folder))

def run_sims(settings_list: list[SimSettings], batch_size: int = -1):
    if batch_size == -1:
        batch_size = multiprocessing.cpu_count()

    processes = []
    args = []
    for settings in settings_list:
        key = str(hash(settings))
        sim_folder  = os.path.join(cache_dir, sims_folder, key )
        sim_data_file = os.path.join(sim_folder, 'sim_data.pkl')

        if not os.path.exists(sim_folder):
            os.mkdir(sim_folder)

        if os.path.exists(sim_data_file):
            settings_list.remove(settings)
            continue

        sim_settings_file = os.path.join(sim_folder, 'sim_settings.json')
        with open(sim_settings_file, 'w') as f:
            json.dump(dataclasses.asdict(settings) , f)

        sim_stdout_file = os.path.join(sim_folder, 'out.txt')

        args.append((settings, sim_data_file, sim_stdout_file))

    print(f"From {len(settings_list)} simulations, {len(settings_list) - len(args)} were already done")
    

    with multiprocessing.Pool(batch_size) as pool:
        pool.starmap(run_sim, args)

    """
    for i in range(0, len(processes), batch_size):
        batch_ti = time.time()
        end_idx = min(i + batch_size, len(processes))
        cur_batch_size = end_idx - i
        for t in processes[i:end_idx]:
            t.start()
        endchar = 's' if cur_batch_size > 1 else ''
        print(f'Executing {cur_batch_size} simulation{endchar}', end='\r')
        for t in processes[i:end_idx]:
            t.join()
        print(f'Batch {(i//batch_size)+1}/{len(processes)//batch_size + (1 if len(processes)%batch_size else 0)} took {time.time() - batch_ti:.2f}s to run')
        batch_ti = time.time()
    """
    print('\nDone')

def run_sim(settings: SimSettings, data_file: os.PathLike, stdout_file: os.PathLike = None):
    if not stdout_file is None:
        sys.stdout = open(stdout_file, 'w')

    sim = SimWrapperTests(settings)
    sim.simulate()
    sim.save_data(data_file, settings=settings)

def fetch_sim_data(settings: SimSettings) -> SimData:
    key = str(hash(settings))
    sim_folder  = os.path.join(cache_dir, sims_folder, key )
    sim_data_file = os.path.join(sim_folder, 'sim_data.pkl')
    if not os.path.exists(sim_data_file):
        print(f'Simulation {key} does not exist')
        return None
    with open(sim_data_file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

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



