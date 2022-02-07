from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sim_settings import SimSettings, def_car_constants, def_controller_parameters
from testing.test_utils import run_sims, fetch_sim_data


if __name__=='__main__':
    # Setup a series of test conditions, trying different deadzone thresholds
    settings_deadzones = [
        SimSettings(
            sim_time=200,
            controller_parameters=def_controller_parameters(
                deadzone_velocity_threshold=val,
            )
        )
        for val in np.arange(0.0, 1, 0.01)
    ]

    run_sims(settings_deadzones)

    # Unpack the data into easier to work with variables
    simulation_data = [fetch_sim_data(s) for s in settings_deadzones]

    thresholds = [s.controller_parameters['deadzone_velocity_threshold'] for s in settings_deadzones]

    time_data = [[instant.time for instant in data.simout] for data in simulation_data]
    energy_data = [[instant.energy_spent for instant in data.simout] for data in simulation_data]

    deadzone_velocity_threshold_values = [data.settings.controller_parameters['deadzone_velocity_threshold'] for data in simulation_data]

    # Setup the plot figures
    plt.figure(0, figsize=(3, 7))

    ax = plt.subplot(111)        
    axes2 = ax.twinx()

    x = []
    line1 = []
    line2 = []
    # For each simulation, plot quantities of interest
    for simulation in range(len(simulation_data)):

        if simulation_data[simulation].collisions != 0:
            print(f"Simulation {simulation} had {simulation_data[simulation].collisions} collisions")
            continue

        velocity_error = simulation_data[simulation].tracking_error_vel
        mean_velocity_error = np.mean(velocity_error)


        time = np.array(time_data[simulation])
        energy = np.array(energy_data[simulation])

        deadzone_velocity_threshold = deadzone_velocity_threshold_values[simulation]
        energy_budget = float(simulation_data[simulation].energy_budget)
        label=f"T={deadzone_velocity_threshold:.2f}, meanVerror={mean_velocity_error:.2f} (E={energy_budget:.0f})"

        line1.append(energy[-1]/1000.0)
        line2.append(time[-1])
        x.append(thresholds[simulation])

    ax.plot(x, line1, 'o', color='C0')
    axes2.plot(x, line2, 'o', color='C1')
    ax.set_ylabel('Energy spent [kJ]', color='C0')
    ax.set_xlabel('Deadzone velocity threshold')
    axes2.set_ylabel('Time [s]', color='C1')

    ax.set_ylim(bottom=0)
    axes2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('deadzone_tests.pdf')
    #plt.show()
