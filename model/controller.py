from audioop import cross
from visualization.utils import State
import numpy as np

class Controller:

    def __init__(self):
        self.current_waypoint = 0
        self.last_position = None
        self.follower = WaypointFollower()

    def output(self, instant, input):
        # Separate input into components
        sensors_output, trajectory_output = input
        current_position = sensors_output[2:4]
        heading = sensors_output[1]

        # TODO: fazer isto bem
        L = 2.2
        # Sum the car's length to the position
        current_position_fixed = current_position + np.array([L * np.cos(heading), L * np.sin(heading)])

        current_waypoint = trajectory_output[self.follower.next_waypoint(trajectory_output[:, 2:4], current_position_fixed)]
        current_error = current_waypoint - sensors_output
        velocity_error = current_error[0]
        current_error = current_error[1:4]

        body_frame_rotation = np.array([[1, 0, 0],
                                        [0, np.cos(heading), np.sin(heading)],
                                        [0, -np.sin(heading), np.cos(heading)]])
        error_body_frame = body_frame_rotation @ current_error

        heading_body_error = np.arctan2(error_body_frame[2], error_body_frame[1]) - sensors_output[4]

        self.force_apply = 100 * velocity_error

        steering_apply = 10 * heading_body_error

        return np.array([self.force_apply, steering_apply])


class WaypointFollower:
    """From a given path, keep a representation of the progress of the car 
    along that path, allowing estimating the current waypoint to follow.
    """
    def __init__(self):
        self.goal = {} # index of current waypoint to follow for each path
    
    def next_waypoint(self, path: np.ndarray, current_position: np.ndarray):
        """Returns the next waypoint to follow for the given path and current position.
        """
        path_b = path.tobytes()
        if path_b not in self.goal: # don't know this path, add it and start following
            self.goal[path_b] = 0

        if self.goal[path_b] is None: # already reached the end of this path
            return None
        
        while self.crossed_goal(path, current_position):
            if self.goal[path_b] == len(path) - 1:
                self.goal[path_b] = None
                return None
            self.goal[path_b] += 1

        return self.goal[path_b]

    def crossed_goal(self, path: np.ndarray, current_position: np.ndarray):
        path_b = path.tobytes()
        if self.goal[path_b] == len(path) - 1: # reached the end of the path
            goals_vec = path[self.goal[path_b]] - path[self.goal[path_b]-1] 
        else:
            goals_vec = path[self.goal[path_b]+1] - path[self.goal[path_b]]
        car_vec = current_position - path[self.goal[path_b]]

        return np.dot(goals_vec, car_vec) > 0

