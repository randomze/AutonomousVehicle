

# trajectory tracking error (3-4 controllers)
# velocity   tracking error (3-4 controllers)
# actuation  (3-4 controllers)


# energy budget bar chart for best controller


import multiprocessing as mp

import matplotlib.pyplot as plt

import examples.ex_different_controllers
import examples.ex_default_controller

def _map(f):
    return f()

if __name__ == "__main__":

    funcs = [
        examples.ex_different_controllers.ex_different_controllers_force,
        examples.ex_different_controllers.ex_different_controllers_steering,
        examples.ex_different_controllers.ex_different_controllers_stability,
        examples.ex_different_controllers.ex_different_controllers_goal_crossing_d,
        examples.ex_default_controller.ex_default_controller_energies,
    ]

    processes = [
        mp.Process(target=_map, args=(f,)) 
        for f in funcs
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
