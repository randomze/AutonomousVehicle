from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

class Controller:

    def __init__(self, gains: dict, goal_crossing_distance: float = -1):
        self.current_waypoint = 0
        self.trajectory = None
        self.last_position = None
        self.follower = WaypointFollower(goal_crossing_threshold=goal_crossing_distance)

        self.gain_force = gains['force']
        self.gain_steering = gains['steering']

        self.lines = []

    def output(self, instant, input):
        # Separate input into components
        sensors_output, trajectory_output = input
        self.trajectory = trajectory_output
        current_position = sensors_output[2:4]
        heading = sensors_output[1]

        # TODO: fazer isto bem
        L = 2.2
        # Sum the car's length to the position
        current_position_fixed = current_position + np.array([L * np.cos(heading), L * np.sin(heading)])

        self.current_waypoint = self.follower.next_waypoint(trajectory_output[:, 2:4], current_position_fixed)
        current_error = trajectory_output[self.current_waypoint] - sensors_output
        velocity_error = current_error[0]
        current_error = current_error[1:4]

        body_frame_rotation = np.array([[1, 0, 0],
                                        [0, np.cos(heading), np.sin(heading)],
                                        [0, -np.sin(heading), np.cos(heading)]])
        error_body_frame = body_frame_rotation @ current_error

        heading_body_error = np.arctan2(error_body_frame[2], error_body_frame[1]) - sensors_output[4]

        self.force_apply = self.gain_force * velocity_error

        steering_apply = self.gain_steering * heading_body_error

        return np.array([self.force_apply, steering_apply])

    def plot(self, ax: plt.Axes, waypoint_window_lims: tuple = (10, 10), cur_color: str = 'r', nei_color: str = 'b', zorder=0):
        """Plot current waypoint and neighbors
        """
        if self.trajectory is None: return

        path = self.trajectory[:, 2:4]
        following_waypoint = path[self.current_waypoint]
        wp_plt = np.array([
            max(self.current_waypoint - waypoint_window_lims[0], 0),
            min(self.current_waypoint + waypoint_window_lims[1], len(path))
        ])

        if len(self.lines) == 0:
            line, = ax.plot(path[wp_plt[0]:wp_plt[1], 0], path[wp_plt[0]:wp_plt[1], 1], color=cur_color, zorder=zorder)
            self.lines.append(line)
            line, = ax.plot(following_waypoint[0], following_waypoint[1], color=nei_color, marker='o', zorder=zorder)
            self.lines.append(line)
            return

        self.lines[0].set_data(path[wp_plt[0]:wp_plt[1], 0], path[wp_plt[0]:wp_plt[1], 1])
        self.lines[1].set_data(following_waypoint[0], following_waypoint[1])

class WaypointFollower:
    """From a given path, keep a representation of the progress of the car 
    along that path, allowing estimating the current waypoint to follow.
    """
    def __init__(self, goal_crossing_threshold: float = 0):
        self.goal = {} # index of current waypoint to follow for each path
        self.goal_crossing_threshold = goal_crossing_threshold
    
    def next_waypoint(self, path: np.ndarray, current_position: np.ndarray):
        """Returns the next waypoint to follow for the given path and current position.
        """
        path_b = path.tobytes()
        if path_b not in self.goal: # don't know this path, add it and start following
            self.goal[path_b] = 0

        while self.crossed_goal(path, current_position):
            if self.goal[path_b] == len(path) - 1:
                return self.goal[path_b]
            self.goal[path_b] += 1

        return self.goal[path_b]

    def crossed_goal(self, path: np.ndarray, current_position: np.ndarray):
        path_b = path.tobytes()
        if self.goal[path_b] == len(path) - 1: # reached the end of the path
            goals_vec = path[self.goal[path_b]] - path[self.goal[path_b]-1] 
        else:
            goals_vec = path[self.goal[path_b]+1] - path[self.goal[path_b]]
        goals_vec = goals_vec / np.linalg.norm(goals_vec)
        car_vec = current_position - path[self.goal[path_b]]

        # project car_vec onto goals_vec
        return np.dot(goals_vec, car_vec) > self.goal_crossing_threshold

