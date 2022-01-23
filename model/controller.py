from audioop import cross
import numpy as np

class Controller:

    def __init__(self):
        self.current_waypoint = 0
        self.last_position = None

    def output(self, instant, input):
        # Separate input into components
        sensors_output, trajectory_output = input
        current_position = sensors_output[2:4]
        heading = sensors_output[1]

        # TODO: fazer isto bem
        L = 2.2

        # Sum the car's length to the position
        current_position_fixed = current_position + np.array([L * np.cos(heading), L * np.sin(heading)])

        # Figure out the crossing line
        if self.current_waypoint == 0:
            waypoint_difference = trajectory_output[self.current_waypoint] - trajectory_output[self.current_waypoint + 1]
        else:
            waypoint_difference = trajectory_output[self.current_waypoint] - trajectory_output[self.current_waypoint - 1]

        # Get only the x,y direction
        waypoint_difference = waypoint_difference[2:4]
        waypoint_difference = waypoint_difference / np.linalg.norm(waypoint_difference)
        # Get the direction of the line orthogonal to the line uniting the two waypoints
        orthogonal_vector = np.array([-waypoint_difference[1], waypoint_difference[0]])
        orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

        # Convert last position and current position into this crossing lines frame
        if not self.last_position is None:
            crossing_frame_matrix = np.array([[waypoint_difference[0], orthogonal_vector[0]],
                                              [waypoint_difference[1], orthogonal_vector[1]]])
            last_position_crossing = crossing_frame_matrix @ (self.last_position - trajectory_output[self.current_waypoint][2:4])
            current_position_crossing = crossing_frame_matrix @ (current_position_fixed - trajectory_output[self.current_waypoint][2:4])

            # Check if between steps the car crossed the waypoint
            if last_position_crossing[0] * current_position_crossing[0] < 0:
                self.current_waypoint += 1

        self.last_position = current_position_fixed

        current_waypoint = trajectory_output[self.current_waypoint]
        current_error = current_waypoint - sensors_output
        velocity_error = current_error[0]
        current_error = current_error[1:4]

        body_frame_rotation = np.array([[1, 0, 0],
                                        [0, np.cos(heading), np.sin(heading)],
                                        [0, -np.sin(heading), np.cos(heading)]])
        error_body_frame = body_frame_rotation @ current_error

        heading_body_error = np.arctan2(error_body_frame[2], error_body_frame[1]) - sensors_output[4]

        force_apply = 1000 * velocity_error

        steering_apply = 10 * heading_body_error

        return np.array([force_apply, steering_apply])