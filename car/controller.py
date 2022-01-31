from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from physics.physics import deadzone


class Controller:
    """ Controls the force applied on the car by the front wheel and the steering angular velocity,
        in order to make the car follow a trajectory of waypoints with reference velocities.
    """

    def __init__(self, params: dict, L, energy_budget: float):
        self.current_waypoint_idx = 0
        self.trajectory = None
        self.last_position = None
        self.L = L

        self.goal_achieved = False

        self.energy_budget = energy_budget

        self.gain_force = params['force']
        self.gain_steering = params['steering']
        self.gain_park = params['park']

        self.deadzone_velocity = params['deadzone_velocity_threshold']
        self.continuous_deadzone = params['deadzone_continuity']
        self.goal_crossing_distance = params['goal_crossing_distance']

        # Create a waypoint follower to keep track of the current waypoint
        self.follower = WaypointFollower(L, goal_crossing_threshold=self.goal_crossing_distance)

        self.lines = []

    def output(self, instant, input):
        """ Returns the control variables as a function of the state of the car with the velocity
        of the center of mass, the trajectory and the energy spent.
        """
        if self.goal_achieved:
            force_apply = 0
            steering_apply = 0
            return (np.array([force_apply, steering_apply]), self.goal_achieved)

        # Separate input into components
        car_state_v_cm, trajectory_output, energy_spent = input
        self.trajectory = trajectory_output
        velocity_cm = car_state_v_cm[0]
        heading = car_state_v_cm[1]
        position_body_frame = car_state_v_cm[2:4]
        phi = car_state_v_cm[4]

        # Sum the car's length to the position
        position_front_wheel = position_body_frame + np.array([self.L * np.cos(heading), self.L * np.sin(heading)])

        target_velocity_cm = trajectory_output[self.current_waypoint_idx][0]
        target_velocity_cm = target_velocity_cm if energy_spent < self.energy_budget else 0

        # Set goal crossing distance as varying linearly with velocity
        max_velocity = trajectory_output[:, 0].max()
        self.follower.goal_crossing_threshold = np.interp(
            velocity_cm, [0, max_velocity],
            [0, self.goal_crossing_distance])

        # Get the waypoint to track
        self.current_waypoint_idx = self.follower.next_waypoint(trajectory_output[:, 2:4], position_front_wheel)

        # Calculate error between car's state and the waypoint's reference state
        current_state_error = trajectory_output[self.current_waypoint_idx] - car_state_v_cm
        velocity_error = current_state_error[0]
        position_error = current_state_error[2:4]

        body_frame_rotation = np.array([[np.cos(heading), np.sin(heading)],
                                        [-np.sin(heading), np.cos(heading)]])
        position_error_body_frame = body_frame_rotation @ position_error

        # Use angle to waypoint as sterring reference
        steering_error = np.arctan2(position_error_body_frame[1], position_error_body_frame[0]) - car_state_v_cm[4]

        steering_apply = self.gain_steering * steering_error
        force_apply = 0
        x_error_body_frame = position_error_body_frame[0]

        # If car is very close to waypoint, do not move steering
        if np.linalg.norm(position_error_body_frame) < self.L + 1e-6:
            steering_apply = 0

        if target_velocity_cm != 0:
            force_apply = self.gain_force * deadzone(velocity_error, self.deadzone_velocity, self.continuous_deadzone)
        else:
            # Stop on waypoint
            velocity_reference = self.gain_park * x_error_body_frame
            force_apply = self.gain_force * (velocity_reference - car_state_v_cm[0])

            # Stop when velocity is sufficiently small
            if velocity_cm < 0.1:
                self.goal_achieved = True

        if energy_spent > self.energy_budget:  # the car ran out of energy
            velocity_error = velocity_cm if velocity_cm > 0 else 0
            force_apply = - self.gain_force * velocity_error

        return (np.array([force_apply, steering_apply]), self.goal_achieved)

    def plot(self, ax: plt.Axes, waypoint_window_lims: tuple = (10, 10),
             cur_color: str = 'r', nei_color: str = 'b', zorder=0):
        """ Plot current waypoint and a line joining the closest waypoints in both directions.
        """
        if self.trajectory is None:
            return

        path = self.trajectory[:, 2:4]
        # Get waypoint that is being followed
        following_waypoint = path[self.current_waypoint_idx]
        # Get closest waypoints
        wp_plt = np.array([
            max(self.current_waypoint_idx - waypoint_window_lims[0], 0),
            min(self.current_waypoint_idx + waypoint_window_lims[1], len(path))
        ])

        # Create lines if they don't exist
        if len(self.lines) == 0:
            line, = ax.plot(path[wp_plt[0]:wp_plt[1], 0], path[wp_plt[0]:wp_plt[1], 1], color=cur_color, zorder=zorder)
            self.lines.append(line)
            line, = ax.plot(following_waypoint[0], following_waypoint[1], color=nei_color, marker='o', zorder=zorder)
            self.lines.append(line)
            return
        # Update lines
        self.lines[0].set_data(path[wp_plt[0]:wp_plt[1], 0], path[wp_plt[0]:wp_plt[1], 1])
        self.lines[1].set_data(following_waypoint[0], following_waypoint[1])


class WaypointFollower:
    """ From a given path, keep a representation of the progress of the car
    along that path, allowing estimating the current waypoint to follow.
    """

    def __init__(self, L, goal_crossing_threshold: float = 0):
        self.goal = {}  # index of current waypoint to follow for each path
        self.goal_crossing_threshold = goal_crossing_threshold
        self.L = L

    def next_waypoint(self, path: np.ndarray, current_position: np.ndarray):
        """ Returns the next waypoint to follow for the given path and current position.
        """
        path_b = path.tobytes()
        if path_b not in self.goal:  # Don't know this path, add it and start following
            self.goal[path_b] = 0

        # Cycle through waypoints that have already been crossed until a waypoint that has not been
        # crossed or until the final is reached
        while self.goal[path_b] != len(path)-1 and self.crossed_goal(path, current_position):
            self.goal[path_b] += 1

        return self.goal[path_b]

    def crossed_goal(self, path: np.ndarray, current_position: np.ndarray):
        """ Returns boolean that represents if the current waypoint has been crossed. 
            Distance to the path is measured as distance to a line that is perpendicular to the path.
        """
        path_b = path.tobytes()
        if self.goal[path_b] == 0:  # First waypoint
            goals_vec = path[self.goal[path_b]+1] - path[self.goal[path_b]]
        else:
            goals_vec = path[self.goal[path_b]] - path[self.goal[path_b]-1]
        goals_vec = goals_vec / np.linalg.norm(goals_vec)
        car_vec = current_position - path[self.goal[path_b]]
        # Note distance to waypoint will be negative if the car hasn't crossed waypoint
        # The goal crossing threshold should be negative as well
        dist = np.dot(goals_vec, car_vec)
        if self.goal[path_b] == len(path)-1:  # Last waypoint
            # Only move past last waypoint if it is crossed
            return dist > 0
        else:
            # Move to next waypoint once car is at a distance of goal_crossing_threshold of the
            # waypoint.
            return dist > self.goal_crossing_threshold
