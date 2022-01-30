from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from model.physics import deadzone


class Controller:

    def __init__(self, params: dict, L, energy_budget: float):
        self.current_waypoint = 0
        self.trajectory = None
        self.last_position = None
        self.L = L

        self.goal_achieved = False

        self.energy_budget = energy_budget

        self.gain_force = params['force']
        self.gain_steering = params['steering']
        self.gain_force_park = params['force_park']

        self.deadzone_velocity = params['deadzone_velocity_threshold']
        self.continuous_deadzone = params['deadzone_continuity']
        self.goal_crossing_distance = params['goal_crossing_distance']

        self.follower = WaypointFollower(L, goal_crossing_threshold=self.goal_crossing_distance)

        self.lines = []

    
    def output(self, instant, input):
        if self.goal_achieved:
            force_apply = 0
            steering_apply = 0
            return (np.array([force_apply, steering_apply]), self.goal_achieved)

        # Separate input into components
        sensors_output, trajectory_output, energy_spent = input
        self.trajectory = trajectory_output
        current_position = sensors_output[2:4]
        heading = sensors_output[1]

        # TODO: fazer isto bem
        L = self.L
        # Sum the car's length to the position
        current_position_fixed = current_position + np.array([L * np.cos(heading), L * np.sin(heading)])

        current_velocity = sensors_output[0]
        target_velocity = trajectory_output[self.current_waypoint][0]
        target_velocity = target_velocity if energy_spent < self.energy_budget else 0

        max_velocity = trajectory_output[:, 0].max()

        self.follower.goal_crossing_threshold = np.interp(current_velocity, [0, max_velocity], [0, self.goal_crossing_distance])


        self.current_waypoint= self.follower.next_waypoint(
            trajectory_output[:, 2:4], current_position_fixed)
            
        current_error = trajectory_output[self.current_waypoint] - sensors_output
        velocity_error = current_error[0]
        current_error = current_error[1:4]

        body_frame_rotation = np.array([[1, 0, 0],
                                        [0, np.cos(heading), np.sin(heading)],
                                        [0, -np.sin(heading), np.cos(heading)]])
        error_body_frame = body_frame_rotation @ current_error

        heading_body_error = np.arctan2(error_body_frame[2], error_body_frame[1]) - sensors_output[4]

        steering_apply = self.gain_steering * heading_body_error
        force_apply = 0
        if target_velocity != 0:
            force_apply = self.gain_force * deadzone(velocity_error, self.deadzone_velocity, self.continuous_deadzone) 
        else:  
            # in final waypoint target velocity is 0, stop on waypoint
            position_error = error_body_frame[1]
            velocity_reference = 0.1 * position_error
            force_apply = self.gain_force * (velocity_reference - sensors_output[0])

            if current_velocity < 0.001:
                self.goal_achieved = True

        if energy_spent > self.energy_budget: # the car ran out of energy
            velocity_error = current_velocity if current_velocity > 0 else 0
            force_apply = - self.gain_force_park * velocity_error


        return (np.array([force_apply, steering_apply]), self.goal_achieved)

    def plot(self, ax: plt.Axes, waypoint_window_lims: tuple = (10, 10),
             cur_color: str = 'r', nei_color: str = 'b', zorder=0):
        """Plot current waypoint and neighbors
        """
        if self.trajectory is None:
            return

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

    def __init__(self, L, goal_crossing_threshold: float = 0):
        self.goal = {}  # index of current waypoint to follow for each path
        self.goal_crossing_threshold = goal_crossing_threshold
        self.L = L

    def next_waypoint(self, path: np.ndarray, current_position: np.ndarray):
        """Returns the next waypoint to follow for the given path and current position.
        """
        path_b = path.tobytes()
        if path_b not in self.goal:  # don't know this path, add it and start following
            self.goal[path_b] = 0

        while self.goal[path_b] != len(path)-1 and self.crossed_goal(path, current_position) :
            self.goal[path_b] += 1

        return self.goal[path_b]

    def crossed_goal(self, path: np.ndarray, current_position: np.ndarray):
        path_b = path.tobytes()
        if self.goal[path_b] == 0:  # first waypoint
            goals_vec = path[self.goal[path_b]+1] - path[self.goal[path_b]]
        else:
            goals_vec = path[self.goal[path_b]] - path[self.goal[path_b]-1]
        goals_vec = goals_vec / np.linalg.norm(goals_vec)
        car_vec = current_position - path[self.goal[path_b]]
        dist = np.dot(goals_vec, car_vec)
        if self.goal[path_b] == len(path)-1:  # last waypoint
            # project car_vec onto goals_vec
            return dist > 0
        else:
            return dist > self.goal_crossing_threshold
