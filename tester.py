from __future__ import annotations
import os
import pickle
from performance.cache_utils import cache_dir
from simulator import SimInstant, SimData, Simulator
from sim_settings import SimSettings, def_car_constants, def_controller_gains

sims_folder = 'sims'
if not os.path.exists(sims_folder):
    os.mkdir(sims_folder)

class SimWrapperTests(Simulator):
    def __init__(self, settings: SimSettings):
        self.settings = settings

        super().__init__(settings.step_size_plot, settings.step_size_sim, settings.car_constants, settings.road_constants, settings.sensor_parameters, settings.controller_gains, settings.traj_endpoints, settings.sim_time, settings.energy_budget, settings.goal_crossing_distance, settings.vis_window, settings.visualization, settings.real_time)

def run_sims(settings_list: list[SimSettings]):
    
    for idx, settings in enumerate(settings_list):
        key = str(hash(settings))
        file = os.path.join(sims_folder, key + '.pkl')

        if os.path.exists(os.path.join(cache_dir, file)):
            print(f'Simulation {idx+1}/{len(settings_list)} already exists')
            continue

        print(f'Running simulation {idx+1}/{len(settings_list)}')
        sim = SimWrapperTests(settings)
        sim.simulate()
        sim.save_data(file)
    print('\nDone')

def fetch_sim_data(settings: SimSettings) -> SimData:
    key = str(hash(settings))
    file = os.path.join(cache_dir, sims_folder, key + '.pkl')
    print(file)
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

    run_sims(settings_lst)

    sim_data = fetch_sim_data(settings_lst[0])

    sims_i_want = [SimSettings(), settings_lst[2]]

    sim_datas_i_want = [fetch_sim_data(s) for s in sims_i_want]



