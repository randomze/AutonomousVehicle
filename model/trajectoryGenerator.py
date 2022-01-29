from __future__ import annotations
from importlib.resources import path
from turtle import position
import numpy as np
import scipy.optimize
import scipy.signal
from environment import road, graph
from visualization.utils import pixel_to_xy, xy_to_pixel


class TrajectoryGenerator:

    def __init__(
            self, road_constants: dict, path: tuple, smoothen_window: int, energy_budget: float,
            vehicle_mass: float, idle_power: float):
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']
        self.energy_budget = energy_budget
        self.vehicle_mass = vehicle_mass
        self.idle_power = idle_power

        road_map, road_graph = road.get_road_info((self.lat, self.lon),
                                                  self.zoom, max_regularization_dist=self.regularization,
                                                  res_zoom_upsample=self.upsampling)

        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph
        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)

        self.path = np.array(self.get_xy_path(*path))
        # Smoothen path in order to avoid sharp turns
        self.path = self.__smoothen_path(self.path, smoothen_window)

        self.thetas = self._calc_waypoint_thetas(positions=self.path)
        # Orientation of each path
        self.p_thetas = self.thetas[1:]
        # Length of each path
        self.lengths = self._calc_path_lengths(positions=self.path)

        self.states = None
        self.goal_states()

        self.last_time_query_idx = 0

    def __smoothen_path(self, path: np.ndarray, smoothen_window: int):
        '''Takes a path and smoothens it using a low pass filter

        path: original path to be smoothened
        smoothen_window: radius of the smoothening low pass filter window

        returns: smoothened path
        '''
        kernel = np.ones((2*smoothen_window + 1, 1)) / (2*smoothen_window + 1)
        smooth_path = scipy.signal.convolve2d(path, kernel, mode='same', boundary='symm')

        return smooth_path

    def get_xy_path(self, init_point, final_point):
        nodes = list(self.graph.get_all_nodes())

        # get first node as the closest to init_point
        init_pixel = xy_to_pixel(*init_point, self.map.shape, self.meters_per_pixel)
        init_node = nodes[np.argmin(
            [np.linalg.norm(np.array(init_pixel) - np.array((node[1], node[0]))) for node in nodes])]

        # get last node as the closest to final_point
        final_pixel = xy_to_pixel(*final_point, self.map.shape, self.meters_per_pixel)
        final_node = nodes[np.argmin(
            [np.linalg.norm(np.array(final_pixel) - np.array((node[1], node[0]))) for node in nodes])]

        # get the path from init_node to final_node
        spt = self.graph.get_spt_dijkstra(init_node)
        nodes_to_traverse = self.graph.get_path_from_spt(final_node, *spt)
        points_to_traverse = [pixel_to_xy((node[1], node[0]), self.map.shape, self.meters_per_pixel)
                              for node in nodes_to_traverse]
        points_to_traverse = points_to_traverse[1:-1]
        points_to_traverse.insert(0, init_point)
        points_to_traverse.append(final_point)

        return points_to_traverse

    def goal_states(self, vel_multiplier: float = 1.0):
        top_maxlim_kmph = 30 * vel_multiplier
        bottom_maxlim_kmph = 7 * vel_multiplier
        top_maxlim = top_maxlim_kmph/3.6
        bottom_maxlim = bottom_maxlim_kmph/3.6
        g = 9.8
        max_deceleration = 0.1*g
        curve_r_to_speed_gain = 10 * vel_multiplier
        E_budget = self.energy_budget

        max_speeds = []

        # For all paths except the last, define a max velocity
        # The final path is attributed with a velocity 0 to signal the controller it should stop at the end of the path
        N = len(self.p_thetas) - 1  # Number of paths to choose a speed for

        # Starts at 0 because loop starts with the path before the last path and the car stops in the last path.
        next_path_v = 0
        for path_i in reversed(range(N)):
            # Limit max velocity based on curvature
            # change of direction
            delta_angle = np.abs(self.p_thetas[path_i+1] - self.p_thetas[path_i])
            # make sure it is the inner angle between the paths
            delta_angle = min(delta_angle, 2*np.pi - delta_angle)
            # convert to a curvature ratio with units: radians of rotation per meter of length of the two paths of the
            # curve
            curve_r = delta_angle / (self.lengths[path_i+1] + self.lengths[path_i])
            # speed limit from curvature of path
            curve_lim = top_maxlim*(1 - curve_r * curve_r_to_speed_gain)

            # max speed must also not require too strong breaking for the next path
            breaking_distance = self.lengths[path_i+1]
            break_lim_squared = next_path_v**2 + 2*max_deceleration*breaking_distance
            break_lim = np.sqrt(break_lim_squared)

            path_max_speed = max(min(curve_lim, break_lim, top_maxlim), bottom_maxlim)

            next_path_v = path_max_speed

            max_speeds = [path_max_speed] + max_speeds

        # Join paths with equal speed limits
        equal_tol = top_maxlim/15
        new_path_lengths = [self.lengths[0]]
        new_max_speeds = [max_speeds[0]]
        # keep track of which new big path the old small ones correspond to
        small_path_indeces = np.empty((N,), dtype=int)
        small_path_indeces[0] = 0
        for i in range(1, N):
            if abs(new_max_speeds[-1] - max_speeds[i]) > equal_tol:
                # keep path
                new_path_lengths += [self.lengths[i]]
                new_max_speeds += [max_speeds[i]]
            else:  # merge path
                new_path_lengths[-1] += self.lengths[i]
            small_path_indeces[i] = len(new_path_lengths)-1

        # optimize speeds on each path to minimise the time spent to reach the start of the final path
        # the final path is for stopping the car, it is not considered in this optimization
        path_lengths = np.array(new_path_lengths)
        max_speeds = np.array(new_max_speeds)
        # set minimum above 0 to avoid division by 0
        min_speeds = np.ones_like(max_speeds) * 1e-6

        v_i = 0     # initial velocity

        # get optimal velocities
        opt_path_vs = self._opt_speeds(v_i, path_lengths, min_speeds, max_speeds,
                                       self.vehicle_mass, self.idle_power, E_budget)

        # recover velocities of small paths
        small_path_vs = opt_path_vs[small_path_indeces]
        # Set velocity of initial waypoint as the same as the second and final as 0
        wp_velocities = np.block([opt_path_vs[0], small_path_vs, 0])

        positions = self.path
        phis = np.zeros(self.path.shape[0])
        # phis_dot = np.zeros(self.path.shape[0])

        self.states = np.column_stack((wp_velocities, self.thetas, positions, phis))

    def output(self, instant):
        return self.states

# def get_entire_distance(positions: np.ndarray):
#     distance = np.zeros(positions.shape[0])

#     for i in range(1, positions.shape[0]):
#         distance[i] = np.linalg.norm(positions[i] - positions[i-1])

    # return np.sum(distance)

    @staticmethod
    def _calc_path_lengths(positions: np.ndarray):
        """ calculates the lengths between waypoints
        """
        distance = np.zeros(positions.shape[0]-1)

        for i in range(0, positions.shape[0]-1):
            distance[i] = np.linalg.norm(positions[i+1] - positions[i])

        return distance

    @staticmethod
    def _calc_waypoint_thetas(positions: np.ndarray):
        """ calculates the theta at waypoint i pointing from waypoint i-1 to waypoint i, except at the first waypoint,
            which points to the second.

            Returns teh theta at each waypoint
        """

        thetas = np.empty(positions.shape[0])

        for i in range(1, positions.shape[0]):
            direction = np.array(positions[i]) - np.array(positions[i-1])
            thetas[i] = np.arctan2(direction[1], direction[0])
        thetas[0] = thetas[1]
        return thetas

    @staticmethod
    def _opt_speeds(v_i, path_lengths, min_speeds, max_speeds, mass, idle_power, E_budget):
        N = len(path_lengths)
        cost_scale = 256

        def travel_time(path_velocities):  # Optimization cost, total time of travel
            path_travel_times = np.divide(path_lengths, path_velocities)
            return path_travel_times.sum()

        def cost(path_velocities):
            return travel_time(path_velocities)/cost_scale

        def jac(path_velocities):  # jacobian of travel time
            return - np.divide(path_lengths, path_velocities**2)/cost_scale

        def total_E_spent(path_velocities):
            v = np.block([v_i, path_velocities])
            v_sqr = v**2
            diff = (v_sqr[1:] - v_sqr[:-1])
            # braking does not recuperate energy
            diff = diff[diff > 0]
            return mass*diff.sum()/2 + travel_time(path_velocities)*idle_power

        # Constraints
        cons = ({'type': 'eq', 'fun': lambda path_velocities: total_E_spent(path_velocities) - E_budget})
        # each path_velocity must respect the min and max limits of both its neighbor paths
        bnds = list(zip(min_speeds, max_speeds))
        # initial guess at optimal velocities
        ini_v = np.ones(N)
        sol = scipy.optimize.minimize(cost, ini_v, method='SLSQP', jac=jac, bounds=bnds,
                                      constraints=cons, options={"maxiter": 2000})
        if not sol.success:
            print(bnds)
            print(sol)
            raise
        # append final velocity for final path
        velocities = np.block([sol.x, 0])
        return velocities


# def get_normalized_times(positions: np.ndarray):
#     times = np.zeros(positions.shape[0])

#     for i in range(2, positions.shape[0]):
#         times[i-1] = times[i-2] + np.linalg.norm(positions[i] - positions[i-1])
#     times[-1] = times[-2]  # gotta figure this one out
#     return times/np.max(times)
