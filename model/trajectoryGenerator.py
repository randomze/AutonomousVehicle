import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
from environment import road, graph
from visualization.utils import pixel_to_xy, xy_to_pixel

class TrajectoryGenerator:

    def __init__(self, road_constants: dict, path: tuple, time: float, smoothen_window: int, energy_budget: float):
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']
        self.energy_budget = energy_budget

        road_map, road_graph = road.get_road_info((self.lat, self.lon), self.zoom, max_regularization_dist=self.regularization, res_zoom_upsample=self.upsampling)

        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph
        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)
        
        self.path = np.array(self.get_xy_path(*path))
        # Smoothen path in order to avoid sharp turns
        self.path = self.__smoothen_path(self.path, smoothen_window)

        self.states = None
        self.goal_states(time)

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
        init_node = nodes[np.argmin([np.linalg.norm(np.array(init_pixel) - np.array((node[1], node[0]))) for node in nodes])]

        # get last node as the closest to final_point
        final_pixel = xy_to_pixel(*final_point, self.map.shape, self.meters_per_pixel)
        final_node = nodes[np.argmin([np.linalg.norm(np.array(final_pixel) - np.array((node[1], node[0]))) for node in nodes])]

        # get the path from init_node to final_node
        mst = self.graph.get_mst_dijkstra(init_node)
        nodes_to_traverse = self.graph.get_path_from_mst(final_node, *mst)
        points_to_traverse = [pixel_to_xy((node[1], node[0]), self.map.shape, self.meters_per_pixel) for node in nodes_to_traverse]
        points_to_traverse = points_to_traverse[1:-1]
        points_to_traverse.insert(0, init_point)
        points_to_traverse.append(final_point)

        return points_to_traverse

    def goal_states(self, total_time: float, vel_multiplier: float = 1.0):
        distances = get_distances(self.path)
        thetas = get_angles(self.path)

        link_length = 10
        max_speed = 30
        min_speed = 7
        E_budget = self.energy_budget

        link_lengths = []
        max_speeds = []
        link_index = 0

        current_distance = 0
        current_curve = 0
        last_angle = 0
        for index in range(self.path.shape[0]):
            current_distance += distances[index]
            current_curve += thetas[index] - last_angle
            last_angle = thetas[index]
            if current_distance > link_length:
                link_lengths += [current_distance]

                current_curve = np.abs(current_curve) % 2*np.pi
                max_speed_current = (max_speed * (2*np.pi - current_curve) + min_speed * current_curve) / (2*np.pi)

                max_speeds += [max_speed_current]
                current_distance = 0
                current_curve = 0

        link_lengths = np.array(link_lengths)
        max_speeds = np.array(max_speeds)
        min_speeds = np.ones_like(max_speeds) * 1e-6
        v_i = 0
        v_f = 0

        M = 810
        N = len(link_lengths)

        def fun(m_v):
            v = np.block([v_i, m_v, v_f])

            avg_v = (v[:-1]+v[1:])/2
            t = np.divide(link_length, avg_v)
            f = t.sum()
            
            # print(f"\tv {v}\n\tavg {avg_v})\n\tf {f}")
            return f

        def total_E_spent(m_v):
            v = np.block([v_i, m_v, v_f])

            v_sqr = v**2
            dif = (v_sqr[1:] - v_sqr[:-1])
            
            # print(dif)
            # braking does not recuperate energy
            dif = dif[dif>0]
            # print(dif)
            return M*dif.sum()/2

        cons = ({'type': 'eq', 'fun': lambda m_v: total_E_spent(m_v) - E_budget})
        # each velocity must respect the min and max limits of both its neighbor paths
        bnds = [(max(min_speeds[i], min_speeds[i+1]), min(max_speeds[i], max_speeds[i+1])) for i in range(N-1)]

        ini_v = np.ones(N-1)

        sol = scipy.optimize.minimize(fun, ini_v, method='SLSQP', bounds=bnds,  constraints=cons)

        velocities = np.empty(self.path.shape[0])
        current_distance = 0
        current_link = 0
        for index, segment in enumerate(distances):
            if index == 0:
                velocities[index] = v_i
            elif current_link == N - 1:
                velocities[index] = v_f
            else:
                velocities[index] = sol.x[current_link]
                current_distance += segment
                if current_distance >= link_lengths[current_link]:
                    current_distance = 0
                    current_link += 1

        positions = self.path
        phis = np.zeros(self.path.shape[0])

        self.states = np.column_stack((velocities, thetas, positions, phis))

    def output(self, instant):
        return self.states

def get_entire_distance(positions: np.ndarray):
    distance = np.zeros(positions.shape[0])

    for i in range(1, positions.shape[0]):
        distance[i] = np.linalg.norm(positions[i] - positions[i-1])

    return np.sum(distance)

def get_distances(positions: np.ndarray):
    distance = np.zeros(positions.shape[0])

    for i in range(1, positions.shape[0]):
        distance[i] = np.linalg.norm(positions[i] - positions[i-1])

    return distance

def get_angles(positions: np.ndarray):
    angles = np.zeros(positions.shape[0])

    for i in range(0, positions.shape[0]-1):
        direction = np.array(positions[i+1]) - np.array(positions[i])
        angle = np.arctan2(direction[1], direction[0])
        angles[i] = angle

    return angles

def get_normalized_times(positions: np.ndarray):
    times = np.zeros(positions.shape[0])

    for i in range(2, positions.shape[0]):
        times[i-1] = times[i-2] + np.linalg.norm(positions[i] - positions[i-1])
    times[-1] = times[-2] # gotta figure this one out
    return times/np.max(times)
