from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from model.physics import CoM_position, MoI
import hashlib


# set default values for mass-related constants
m = 3
n = 2
com_r, com_delta = CoM_position(m, n)
Izz = MoI(m, n)

def def_car_constants(L=2.2, Lr=0.566, Lf=0.566, d=0.64, r=0.256, Length=3.332, Width=1.508, M=810.0, Izz=Izz, com_r=com_r, com_delta=com_delta, wheel_width=0.1, idle_power=1):
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

def def_road_constants(lat=38.7367256, lon=-9.1388871, zoom=16, upsampling=3, regularization=5):
    return {
        'lat': lat,
        'lon': lon,
        'zoom': zoom,
        'upsampling': upsampling,
        'regularization': regularization,
    }

def def_sensor_parameters():
    return {}

def def_controller_gains(force=1000, force_park=10, steering=100):
    return {
        'force': force,
        'force_park': force_park,
        'steering': steering,
    }


@dataclass(frozen=True)
class SimSettings:
    # general simulation settings
    step_size_plot: float = 0.1
    step_size_sim: float = 0.01
    sim_time: float = 100

    car_constants: Dict = field(default_factory=def_car_constants) 
    road_constants: Dict = field(default_factory=def_road_constants)
    sensor_parameters: Dict = field(default_factory=def_sensor_parameters)
    controller_gains: Dict = field(default_factory=def_controller_gains)
    traj_endpoints: tuple =((-5, 10), (-85, 100))
    energy_budget: float = 10000
    goal_crossing_distance: float = -2.54

    # visualization
    vis_window: tuple = ((-20, 20), (-20, 20))
    visualization: bool = False
    real_time: bool = False

    def __hash__(self) -> int:
        id = str(self)
        return int(hashlib.sha1(id.encode('utf-8')).hexdigest(), base=16)