# Robotics Laboratory 2

This software was developed for the 2nd Laboratory of the Robotics Course at IST. The software developed
can be roughly separated into 3 parts:

- The simulator, which is a simple simulator with a fixed discrete time step. The modeled modules are
defined as continuous time dynamical systems. Thus they expose two important functions, the derivative function,
which relates the derivative of the state with the current state and inputs, and the output function,
which relates the output with the current state (For simplicity of the simulation the output was not allowed
to depend on the current input. This did not cause issues since these systems naturally already had this 
property). Through the choice of an adequate time step for the simulation this method proved sufficient
and did not lead to numerical stability problems.
- The autonomous car system, which is composed by various components: the car model, the path planner,
the trajectory generator and the controller.
- The testing suite which both allows validation of the rest of the software and the determination of
parameters for the controller, by testing various combinations of the these parameteres and optimizing
for a certain cost function.

## Running the Simulator

Running the simulator allows the user to see a real-time simulation of the car in an environment which may
specified by the user, with a complete graphical interface which shows both the movement of the car as
the remaining interest variables: current velocity, energy and number of colisions.

To run the simulator, the user must merely run the command:

```
python3 simulator.py
```

The simulator also produces a `simulation.mp4` video file so the simulation can be seen again afterwards.

### Choosing different settings for the simulator

The simulator settings are all gathered in a SimSettings class, details of which can be altered in its instantiation in simulator.py .The parameter list is quite extensive and can be fully seen in sim_settings.py, accounting for each configurable element of the simulation.

## Running the Testing Suite

The testing suite is a module with 3 important submodules:

- The controller tests, which produce error plots for a variety of controllers, including the default
one.
- The deadzone tests, which produce a series of plots of energy spent vs time, to evaluate the impact
of the controller velocity deadzone in diminishing the energy spent while simultaneously tracking
the impact in the error. No simulations with collisions are admissible and they are not counted.
- The controller parameters mass tests, which runs a very long simulation (~8 hours) and takes up
a lot of space due to the cache (~30Gb). This runs simulations that test all combinations of the important
controller parameters (the goal crossing distance, the velocity control gain and the steering control gain),
each within a discrete number of options. Each of these controllers is tested on a set of different trajectories,
with different path characteristics (sharper turns, larger straight stretches, etc.), then measuring a set
of relevant metrics, like the average tracking error, average velocity error, etc. These are then used to determine
a cost for each controller. The controller that minimizes this cost function is considered the optimal
controller. The default controller for the simulator is the one determined through this mean.

To run each of these tests, the user must run the following command:
```
python3 -m testing.[controller_tests/deadzone_tests/mass_tester]
```

## Requirements

In order to run this software, some external python libraries are required. To install them
simply run:
```
pip3 install -r requirements.txt
```