from __future__ import annotations
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from simulator import SimData
from sim_settings import TrajectoryPreset


from testing.mass_tester import SimBundle, test_bundles, get_result
from testing.test_utils import run_sims, fetch_sim_data


def sim_and_plot(settings, **plot_kwargs):
    run_sims(settings)
    data = [fetch_sim_data(setting) for setting in settings]
    plot_sims(data, **plot_kwargs)

def sim_and_plot_bundle_energies(settings: SimBundle, **plot_kwargs):
    test_bundles(settings)
    data = [get_result(setting) for setting in settings]
    plot_sims_bundle(data, **plot_kwargs)


def time_to_distance(time: np.ndarray, car_velocities: np.ndarray):
    """ Convert simulation time into distance traveled using the car's velocity.
    """
    time_step = time[1] - time[0]

    distance = np.empty_like(time)
    distance[0] = 0

    for index in range(time.shape[0] - 1):
        distance[index + 1] = distance[index] + car_velocities[index] * time_step

    return distance

def plot_sims(sims: list[SimData], use_time_as_x: bool = False, labels: tuple[str] = None, data_folder: str = None):
    """ Plot the joint tracking errors for position and velocity and actuation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    time = []
    for sim in sims:
        time.append(np.array([instant.time for instant in sim.simout]))
    x = []
    if use_time_as_x:
        x = time
        xlabel = 'Time [s]'
    else:
        for idx, sim in enumerate(sims):
            velocities = np.array([instant.car_state_v_cm[0] for instant in sim.simout])
            x.append(time_to_distance(time[idx], velocities))

        xlabel = 'Distance Along Path [m]'

    collisions = [sim.collisions for sim in sims]
    energy_budget = [sim.energy_budget for sim in sims]
    energy_used = [sim.simout[-1].energy_spent for sim in sims]

    actuation_force = []
    actuation_steering = []
    for sim in sims:
        actuation_force.append([instant.controller_actuation[0] for instant in sim.simout])
        actuation_steering.append([instant.controller_actuation[1] for instant in sim.simout])


    for idx, sim in enumerate(sims):
        color = 'C' + str(idx)
        line_style = 'solid' if idx == 0 else 'dashed'
        line_thickness = 1.5 if idx == 0 else 0.8
        axes[0][0].plot(x[idx], sim.tracking_error_pos, color=color, linestyle=line_style, linewidth=line_thickness)
        axes[1][0].plot(x[idx], sim.tracking_error_vel, color=color, linestyle=line_style, linewidth=line_thickness)
        axes[0][1].plot(x[idx], actuation_force[idx], color=color, linestyle=line_style, linewidth=line_thickness)
        axes[1][1].plot(x[idx], actuation_steering[idx], color=color, linestyle=line_style, linewidth=line_thickness)
        if idx == 0:
            ylim = axes[1][1].get_ylim()
        else:
            ylim_now = axes[1][1].get_ylim()
            if ylim_now[1] > 40:
                axes[1][1].set_ylim(ylim)
        
    axes[0][0].set_ylabel('Position Error [m]')
    axes[1][0].set_ylabel('Velocity Error [m/s]')
    axes[0][1].set_ylabel('Velocity Actuation [N]')
    axes[1][1].set_ylabel('Steering Actuation [rad/s]')

    axes[1][1].legend(labels)

    axes[1][0].set_xlabel(xlabel)
    axes[1][1].set_xlabel(xlabel)
    plt.tight_layout()

    plt.savefig(data_folder+".pdf")

    # save collisions, energy_budget, energy_used in json
    dict_info = {}
    for sim_idx in range(len(sims)):
        dict_info[f'Controller {sim_idx+1}'] = {
            'Collisions': collisions[sim_idx],
            'Energy Used': energy_used[sim_idx],
            'Energy Budget': energy_budget[sim_idx],
        }
    with open(data_folder + '.json', 'w') as f:
        json.dump(dict_info, f, indent=4)


def plot_sims_bundle(bundles: list[list[SimData]], data_folder: str = None, labels: tuple[str] = None):

    energy_used_ratio = []
    for sims in bundles:
        energy_used_ratio.append([sim.simout[-1].energy_spent / sim.energy_budget for sim in sims])
    trajectories = [[sim.settings.traj_endpoints for sim in sims] for sims in bundles]
    trajectory_names = [[TrajectoryPreset(traj).name for traj in bundle_traj] for bundle_traj in trajectories]

    # make a bar chart of energy_used_ratio for each trajectory
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.3
    N = len(bundles)
    x = np.arange(len(trajectory_names[0]))
    for idx in range(len(energy_used_ratio)):
        ax.bar(x + width*idx, energy_used_ratio[idx], width)

    ax.set_ylabel('Energy Used / Energy Budget')
    ax.set_xlabel('Trajectory')
    ax.set_xticks(x + width*(N//2), trajectory_names[0])
    if labels is not None:
        ax.legend(labels)
    plt.tight_layout()

    plt.savefig(data_folder+".pdf")
