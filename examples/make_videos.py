import os

from simulator import Simulator

from sim_settings import SimSettings, def_controller_parameters, TrajectoryPreset

import matplotlib.pyplot as plt



if __name__ == '__main__':
    videos_folder = 'videos'
    if not os.path.isdir(videos_folder):
        os.mkdir(videos_folder)

    settings_name = []
    settings_name += [(
        SimSettings(
            visualization=True,
            step_size_plot=1/60,
            traj_endpoints=TrajectoryPreset.Balanced1.value,
        ),
        'default_controller_balanced1'
    )]

    settings_name += [(
        SimSettings(
            visualization=True,
            step_size_plot=1/60,
            traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
        ),
        'default_controller_verysharpturn'
    )]

    settings_name += [(
        SimSettings(
            visualization=True,
            step_size_plot=1/60,
            traj_endpoints=TrajectoryPreset.VerySharpTurn.value,
            controller_parameters=def_controller_parameters(
                steering=1.0,
            ),
        ),
        'low_ks_controller_verysharpturn'
    )]

    settings_name += [(
        SimSettings(
            visualization=True,
            step_size_plot=1/60,
            traj_endpoints=TrajectoryPreset.Corners.value,
        ),
        'default_controller_corners'
    )]

    for settings, name in settings_name:
        sim = Simulator(settings)
        sim.simulate()
        sim.save_data(video_file=os.path.join(videos_folder, f'{name}.mp4'))
    