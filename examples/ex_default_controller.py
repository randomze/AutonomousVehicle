from __future__ import annotations

from examples.plot_primitives import sim_and_plot_bundle_energies
from testing.mass_tester import make_trajectories_bundle
from sim_settings import SimSettings, def_controller_parameters

def ex_default_controller_energies():
    settings = [
        SimSettings(),
        SimSettings(controller_parameters=def_controller_parameters(
            deadzone_velocity_threshold=0.0,
        )),
        SimSettings(controller_parameters=def_controller_parameters(
            deadzone_velocity_threshold=0.5,
        )),
    ]

    bundles = [make_trajectories_bundle(setting) for setting in settings]

    labels = [
        '$d_v = 0.2$ (default)',
        '$d_v = 0.0$',
        '$d_v = 0.5$',
    ]

    sim_and_plot_bundle_energies(bundles, data_folder='ex_different_deadzones', labels=labels)

