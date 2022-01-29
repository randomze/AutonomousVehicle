from __future__ import annotations
import os
import pickle
import multiprocessing 
import time

import numpy as np
from performance.cache_utils import cache_dir
from simulator import SimInstant, SimData, Simulator
from sim_settings import SimSettings, def_car_constants, def_controller_gains

sims_folder = 'sims'
if not os.path.exists(os.path.join(cache_dir, sims_folder)):
    os.mkdir(os.path.join(cache_dir,sims_folder))

class SimWrapperTests(Simulator):
    def __init__(self, settings: SimSettings):
        self.settings = settings

        super().__init__(settings.step_size_plot, settings.step_size_sim, settings.car_constants, settings.road_constants, settings.sensor_parameters, settings.controller_gains, settings.traj_endpoints, settings.sim_time, settings.energy_budget, settings.goal_crossing_distance, settings.vis_window, settings.visualization, settings.real_time)

def run_sims(settings_list: list[SimSettings], batch_size: int = -1):
    if batch_size == -1:
        batch_size = multiprocessing.cpu_count()

    processes = []
    for settings in settings_list:
        t = multiprocessing.Process(target=run_sim, args=(settings,))
        processes.append(t)
    
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
        print(f'Batch {(i//batch_size)+1}/{len(processes)//batch_size + len(processes)%batch_size} took {time.time() - batch_ti:.2f}s to run')
        batch_ti = time.time()
        
    print('\nDone')

def run_sim(settings: SimSettings):
    key = str(hash(settings))
    file = os.path.join(sims_folder, key + '.pkl')

    if os.path.exists(os.path.join(cache_dir, file)):
        return

    sim = SimWrapperTests(settings)
    sim.simulate()
    sim.save_data(file, settings=settings)

def fetch_sim_data(settings: SimSettings) -> SimData:
    key = str(hash(settings))
    file = os.path.join(cache_dir, sims_folder, key + '.pkl')
    if not os.path.exists(file):
        print(f'Simulation {key} does not exist')
        return None
    with open(file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    settings_lst = [
        SimSettings(), # default settings
        SimSettings(car_constants=def_car_constants(idle_power=10)),
        SimSettings(controller_gains=def_controller_gains(steering=120))
    ]


    settings_lst += [SimSettings(goal_crossing_distance=val) for val in np.arange(-3.5, -0.5, 0.1)]

    run_sims(settings_lst)

    sim_data = fetch_sim_data(settings_lst[0])

    sims_i_want = [SimSettings(), settings_lst[2]]

    sim_datas_i_want = [fetch_sim_data(s) for s in sims_i_want]

    steering_vals = [d.settings.controller_gains['steering'] for d in sim_datas_i_want]

    print(steering_vals)



