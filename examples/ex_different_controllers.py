from __future__ import annotations

from sim_settings import SimSettings, TrajectoryPreset, def_controller_parameters
from examples.plot_primitives import sim_and_plot

def ex_different_controllers_force():
    controller_1_settings = SimSettings( # default controller
        controller_parameters=def_controller_parameters(
        ),
    )
    controller_2_settings = SimSettings( 
        controller_parameters=def_controller_parameters(
            force=200.0,
        ),
    )
    controller_3_settings = SimSettings(
        controller_parameters=def_controller_parameters(
            force=2000.0,
        ),
    )

    settings = [
        controller_1_settings,
        controller_2_settings,
        controller_3_settings,
    ]

    labels = [
        '$(K_v, K_s) = (733.33, 15.21)$ (default)',
        '$(K_v, K_s) = (200.00, 15.21)$',
        '$(K_v, K_s) = (2000.0, 15.21)$',
    ]

    sim_and_plot(settings, labels=labels, data_folder='ex_different_controllers_force')

    

def ex_different_controllers_steering():
    controller_1_settings = SimSettings( # default controller
        controller_parameters=def_controller_parameters(
        ),
    )
    controller_2_settings = SimSettings( 
        controller_parameters=def_controller_parameters(
            steering=1.0,
        ),
    )
    controller_3_settings = SimSettings(
        controller_parameters=def_controller_parameters(
            steering=150.0,
        ),
    )

    settings = [
        controller_1_settings,
        controller_2_settings,
        controller_3_settings,
    ]

    labels = [
        '$(K_v, K_s) = (733.33, 15.21)$ (default)',
        '$(K_v, K_s) = (733.33, 1.00)$',
        '$(K_v, K_s) = (733.33, 150.00)$',
    ]

    sim_and_plot(settings, labels=labels, data_folder='ex_different_controllers_steering')

    

def ex_different_controllers_goal_crossing_d():
    controller_1_settings = SimSettings( # default controller
        controller_parameters=def_controller_parameters(
        ),
        traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
    )
    controller_2_settings = SimSettings( 
        controller_parameters=def_controller_parameters(
            goal_crossing_distance=-1.0,
        ),
        traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
    )
    controller_3_settings = SimSettings(
        controller_parameters=def_controller_parameters(
            goal_crossing_distance=-3.5,
        ),
        traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
    )

    settings = [
        controller_1_settings,
        controller_2_settings,
        controller_3_settings,
    ]

    labels = [
        '$d_g = -2.54$ (default)',
        '$d_g = -1.00$',
        '$d_g = -3.50$',
    ]

    sim_and_plot(settings, labels=labels, data_folder='ex_different_controllers_goal_crossing_d')



def ex_different_controllers_stability():
    controller_1_settings = SimSettings( # default controller
        controller_parameters=def_controller_parameters(
        ),
    )
    controller_2_settings = SimSettings( 
        controller_parameters=def_controller_parameters(
            force=180000.0,
        ),
    )

    settings = [
        controller_1_settings,
        controller_2_settings,
    ]

    labels = [
        '$(K_v, K_s) = (733.33, 15.21)$ (default)',
        '$(K_v, K_s) = (180000.0, 1.00)$',
    ]

    sim_and_plot(settings, labels=labels, use_time_as_x=True, data_folder='ex_different_controllers_stability')


