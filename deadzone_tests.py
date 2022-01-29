import matplotlib.pyplot as plt
import numpy as np

from tester import run_sims, fetch_sim_data
from sim_settings import SimSettings, def_car_constants, def_controller_gains


if __name__=='__main__':
    settings_deadzones = [
        SimSettings(
            sim_time=200,
            controller_gains=def_controller_gains(
                deadzone_continuity=True,
                deadzone_velocity_threshold=val
            )
        )
        for val in np.arange(0.0, 5, 0.1)
    ]

    run_sims(settings_deadzones)

    simulation_data = [fetch_sim_data(s) for s in settings_deadzones]

    time_data = [[instant.time for instant in data.simout] for data in simulation_data]
    energy_data = [[instant.energy_spent for instant in data.simout] for data in simulation_data]

    car_state_data = [[instant.car_state_v_cm for instant in data.simout] for data in simulation_data]
    reference_data = [[instant.controller_reference for instant in data.simout] for data in simulation_data]

    deadzone_velocity_threshold_values = [data.settings.controller_gains['deadzone_velocity_threshold'] for data in simulation_data]
    energy_budget_values = [data.settings.energy_budget for data in simulation_data]

    plt.figure(0)
    plt.title("Energy spent vs. time for different velocity deadzone thresholds")
    for simulation in range(len(simulation_data)):
        car_state = np.vstack(car_state_data[simulation])
        reference = np.vstack(reference_data[simulation])

        car_velocities = car_state[:, 0]
        reference_velocities = reference[:, 0]

        velocity_error = reference_velocities - car_velocities
        mean_velocity_error = np.mean(velocity_error)


        time = np.array(time_data[simulation])
        energy = np.array(energy_data[simulation])


        deadzone_velocity_threshold = deadzone_velocity_threshold_values[simulation]
        energy_budget = energy_budget_values[simulation]

        plt.plot(time, energy, label=f"T={deadzone_velocity_threshold:.2f}, meanVerror={mean_velocity_error:.2f} (E={energy_budget:.0f})")

    plt.legend(loc="upper left")
    plt.show()
