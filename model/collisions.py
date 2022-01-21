import numpy as np


def is_colliding(obj1, obj2):
    """Detects if two objects are colliding by the separating axis theorem.
    
    Objects are represented by their vertices (anti-clockwise).
    Args:
        obj1 (np.ndarray): The first object.
        obj2 (np.ndarray): The second object.

    Returns:
        bool: True if colliding, False otherwise.

    """
    if not is_colliding_ax1(obj1, obj2):
        return False
    if not is_colliding_ax1(obj2, obj1):
        return False

    return True

def is_colliding_ax1(obj1: np.ndarray, obj2: np.ndarray):

    for i in range(len(obj1)):
        # vertexes
        x1, y1 = obj1[i]
        x2, y2 = obj1[(i+1) % len(obj1)]

        # edge
        dx, dy = np.array([x2-x1, y2-y1])
        # normal (unit)
        proj_axis = np.array([-dy, dx])/np.linalg.norm([dx, dy])
        
        # project obj1 and obj2 vertices onto the normal
        proj_obj1 = np.dot(obj1, proj_axis)
        proj_obj2 = np.dot(obj2, proj_axis)
        
        # find the min and max of the projections
        p1m, p1M = np.min(proj_obj1), np.max(proj_obj1)
        p2m, p2M = np.min(proj_obj2), np.max(proj_obj2)

        # if there is no overlap, return false
        if p2m > p1M or p1m > p2M:
            return False
    return True

