from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Dict

from physics.physics import center_of_mass_position, moment_of_inertia


# Set default values for mass-related constants
m = 3
n = 2
com_r, com_delta = center_of_mass_position(m, n)
Izz = moment_of_inertia(m, n)

def def_car_constants(L=2.2, Lr=0.566, Lf=0.566, d=0.64, r=0.256, Length=3.332, Width=1.508, M=810.0, Izz=Izz, com_r=com_r, com_delta=com_delta, wheel_width=0.1, idle_power=500):
    return {
        'L': L,
        'Lr': Lr,
        'Lf': Lf,
        'd': d,
        'r': r,
        'Length': Length,
        'Width': Width,
        'M': M,
        'Izz': Izz,
        'r_cm': com_r,
        'delta_cm': com_delta,
        'wheel_width': wheel_width,
        'idle_power' : idle_power
    }

def def_road_constants(lat=38.7367256, lon=-9.1388871, zoom=16, upsampling=3, 
                    regularization=5, smoothen_window=5, speed_limit_discretization_N=10,
                    max_top_speed_kmh=30, min_top_speed_kmh=10, max_deceleration_g_ratio=0.1,
                    max_optimizer_iterations=3000, energy_estimation_multiplier=1.5,
                    energy_reserve_ratio=0.3):
    return {
        'lat': lat,
        'lon': lon,
        'zoom': zoom,
        'upsampling': upsampling,
        'regularization': regularization,
        'trajectory_smoothen_window': smoothen_window,
        'speed_limit_discretization_N': speed_limit_discretization_N,
        'max_top_speed_kmh': max_top_speed_kmh,
        'min_top_speed_kmh': min_top_speed_kmh,
        'max_deceleration_g_ratio': max_deceleration_g_ratio,
        'max_optimizer_iterations': max_optimizer_iterations,
        'energy_estimation_multiplier': energy_estimation_multiplier,
        'energy_reserve_ratio': energy_reserve_ratio,
    }

def def_controller_parameters(force=733.33, steering=15.21, deadzone_velocity_threshold=0.2, deadzone_continuity: bool = True, goal_crossing_distance=-2.54, park=0.1):
    return {
        'force': force,
        'steering': steering,
        'park': park,
        'deadzone_velocity_threshold': deadzone_velocity_threshold,
        'deadzone_continuity': deadzone_continuity,
        'goal_crossing_distance': goal_crossing_distance,
    }

class TrajectoryPreset(Enum):
    Corners = ((-290, 340), (-5, 300))
    SharpTurns = ((-30, -330), (120, -330))
    VerySharpTurn = ((-103, 135), (-103, 144))
    StraightWide = ((180, -250), (170, 355))
    StraightNarrow = ((-210, -365), (-323, 353))
    Balanced1 = ((-230, -205), (125, -3))
    Balanced2 = ((40, 185), (75, 45))
    Balanced3 = ((303, -343), (275, -343))


@dataclass
class SimSettings:
    # General simulation settings
    step_size_plot: float = 0.1
    step_size_sim: float = 0.01
    sim_time: float = 100

    sim_time_after_stop: float = 5

    car_constants: Dict = field(default_factory=def_car_constants) 
    road_constants: Dict = field(default_factory=def_road_constants)
    controller_parameters: Dict = field(default_factory=def_controller_parameters)
    traj_endpoints: tuple = TrajectoryPreset.Balanced1.value
    
    # The energy budget is defined as a Tuple. If the first element is not None, then it is the 
    # energy budget. This is useful in a real scenario, where the car might be constrained by
    # the battery's capacity or the fuel tank's size. In the simulation setting, the first element
    # is set to None and the second element is set to the maximum velocity, from which the energy
    # budget is estimated.
    energy_budget: tuple = (None, 10)

    # Visualization
    vis_window: tuple = ((-20, 20), (-20, 20))
    visualization: bool = False
    view_while_sim: bool = False

    def __hash__(self) -> int:
        id = str(self)
        return int(hashlib.sha1(id.encode('utf-8')).hexdigest(), base=16)