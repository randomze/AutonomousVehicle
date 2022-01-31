from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from testing.test_utils import run_sims, fetch_sim_data
from sim_settings import SimSettings, def_car_constants, def_controller_parameters


if __name__=='__main__':
    settings_deadzones = [
        SimSettings(
            sim_time=200,
            controller_parameters=def_controller_parameters(
                deadzone_velocity_threshold=val,
                steering=115.52,
                force=733.33,
                goal_crossing_distance=-2.0,
            )
        )
        for val in np.arange(0.0, 0.5, 0.01)
    ]
    settings_deadzones += [
        SimSettings(
            sim_time=200,
            controller_parameters=def_controller_parameters(
                deadzone_velocity_threshold=val,
                steering=115.5,
                force=733.33,
                goal_crossing_distance=-2.0,
            )
        )
        for val in np.arange(0.5, 5, 0.5)
    ]
    settings_deadzones += [
        SimSettings(
            sim_time=200,
            controller_parameters=def_controller_parameters(
                deadzone_velocity_threshold=val,
                steering=115.5,
                force=733.33,
                goal_crossing_distance=-2.0,
            )
        )
        for val in np.arange(0.5, 3, 0.1)
    ]

    run_sims(settings_deadzones)

    simulation_data = [fetch_sim_data(s) for s in settings_deadzones]

    thresholds = [s.controller_parameters['deadzone_velocity_threshold'] for s in settings_deadzones]

    time_data = [[instant.time for instant in data.simout] for data in simulation_data]
    energy_data = [[instant.energy_spent for instant in data.simout] for data in simulation_data]


    deadzone_velocity_threshold_values = [data.settings.controller_parameters['deadzone_velocity_threshold'] for data in simulation_data]
    energy_budget_values = [data.settings.energy_budget for data in simulation_data]

    plt.figure(0)
    plt.title("Energy spent vs. time for different velocity deadzone thresholds")

    ax = plt.subplot(111)        
    axes2 = ax.twinx()

    x = []
    line1 = []
    line2 = []
    for simulation in range(len(simulation_data)):

        if simulation_data[simulation].collisions != 0:
            print(f"Simulation {simulation} had {simulation_data[simulation].collisions} collisions")
            continue

        velocity_error = simulation_data[simulation].tracking_error_vel
        mean_velocity_error = np.mean(velocity_error)


        time = np.array(time_data[simulation])
        energy = np.array(energy_data[simulation])

        deadzone_velocity_threshold = deadzone_velocity_threshold_values[simulation]
        energy_budget = float(simulation_data[simulation].settings.energy_budget[0])
        label=f"T={deadzone_velocity_threshold:.2f}, meanVerror={mean_velocity_error:.2f} (E={energy_budget:.0f})"

        line1.append(energy[-1])
        line2.append(time[-1])
        x.append(thresholds[simulation])

    ax.plot(x, line1, 'o', color='C0')
    axes2.plot(x, line2, 'o', color='C1')
    ax.set_ylabel('Energy spent', color='C0')
    ax.set_xlabel('Deadzone velocity threshold')
    axes2.set_ylabel('Time', color='C1')


    plt.show()
