from __future__ import annotations
from typing import Tuple
import numpy as np

def CoM_position(m: int, n: int) -> Tuple:
    d = 0.64
    W = 2 * d
    Lm = 2.2
    Lr = Lf = 0.566
    L = Lr + Lm + Lf

    com_x = 0
    for j in range(n):
        com_x += W/2 - (j - 1/2)*W/n
    com_x = com_x / n

    com_y = 0
    for i in range(m):
        com_y += L/2 - (i - 1/2)*L/m
    com_y = com_y / m

    com_r = np.sqrt((com_x/Lm) ** 2 + (com_y/Lm)**2)
    com_delta = np.arctan2(com_x, com_y)

    return (com_r, com_delta)

def MoI(m: int, n: int) -> float:
    d = 0.64
    W = 2 * d
    Lm = 2.2
    Lr = Lf = 0.566
    L = Lr + Lm + Lf
    M = 810

    Izz = 0
    for i in range(m):
        for j in range(n):
            Izz += (W/2 - (j - 1/2)*W/n)**2 + (L/2 - (i - 1/2)*L/m)**2

    Izz *= M/(m*n)

    return Izz
