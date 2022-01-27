from __future__ import annotations
from simulator import SimInstant, Simulator
from sim_settings import SimSettings

class SimWrapperTests(Simulator):
    def __init__(self, settings: SimSettings):
        self.settings = settings

        super().__init__(settings.step_size_plot, settings.step_size_sim, settings.car_constants, settings.road_constants, settings.sensor_parameters, settings.controller_gains, settings.traj_endpoints, settings.sim_time, settings.energy_budget, settings.goal_crossing_distance, settings.vis_window, settings.visualization, settings.real_time)

def run_sims(settings_list: list[SimSettings]):
    
    for settings in settings_list:
        sim = SimWrapperTests(settings)
        sim.simulate()
        sim.save_data(str(hash(settings)) + '.pkl')

if __name__ == '__main__':

    settings = SimSettings()
    sim = SimWrapperTests(settings)
    sim.simulate()


