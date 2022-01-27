import matplotlib.pyplot as plt
import numpy as np

from tester import run_sims, fetch_sim_data
from sim_settings import SimSettings, def_car_constants, def_controller_gains

if __name__=='__main__':
    settings_list = [
        SimSettings(controller_gains=def_controller_gains(force=10)),
        SimSettings(controller_gains=def_controller_gains(force=100)),
        SimSettings(controller_gains=def_controller_gains(force=1000)),
        SimSettings(controller_gains=def_controller_gains(steering=1)),
        SimSettings(controller_gains=def_controller_gains(steering=10)),
        SimSettings(controller_gains=def_controller_gains(steering=100))
    ]

    run_sims(settings_list)

    simulation_data = [fetch_sim_data(s) for s in settings_list]

    car_state_data = [[instant.car_state_v_cm for instant in data.simout] for data in simulation_data]
    reference_data = [[instant.controller_reference for instant in data.simout] for data in simulation_data]
    time_data = [[instant.time for instant in data.simout] for data in simulation_data]

    controller_gains = [(data.settings.controller_gains['force'], data.settings.controller_gains['steering']) for data in simulation_data]

    for simulation in range(len(simulation_data)):
        car_state = np.vstack(car_state_data[simulation])
        reference = np.vstack(reference_data[simulation])
        time = np.array(time_data[simulation])

        car_velocities = car_state[:, 0]
        reference_velocities = reference[:, 0]

        velocity_error = reference_velocities - car_velocities
        force_gain, steering_gain = controller_gains[simulation]
        plt.figure(0)
        plt.plot(time, velocity_error, label=f"K_f={force_gain}, K_s={steering_gain}")

    plt.legend(loc="upper right")
    plt.show()