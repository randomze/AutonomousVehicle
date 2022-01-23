import numpy as np
import matplotlib.pyplot as plt
from environment import road, graph
from visualization.utils import pixel_to_xy, xy_to_pixel, figure_number

class TrajectoryGenerator:

    def __init__(self, road_constants: dict, path: tuple, time: float):
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']

        road_map, road_graph = road.get_road_info((self.lat, self.lon), self.zoom, max_regularization_dist=self.regularization, res_zoom_upsample=self.upsampling)

        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph
        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)
        
        self.path = np.array(self.get_xy_path(*path))
        self.goal = None
        self.states = None
        self.times = None
        self.goal_states(time)

        self.last_time_query_idx = 0


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
        times = get_normalized_times(self.path)*total_time
        velocities = vel_multiplier*np.ones(times.shape[0])*get_entire_distance(self.path)/total_time
        thetas = get_angles(self.path)
        positions = self.path
        phis = np.zeros(self.path.shape[0])

        self.times = times
        self.states = np.column_stack((velocities, thetas, positions, phis))


    def plot(self, clf: bool = False, block: bool = False, color ='r', cur_color = 'b'):
        plt.figure(figure_number)
        if clf: plt.clf()
        for point in self.path:
            plt.scatter(point[0], point[1], color=color)

        if self.goal is not None:
            plt.scatter(self.goal[0], self.goal[1], color=cur_color)
        plt.show(block=block)

    def output(self, instant):
        if instant > self.times[self.last_time_query_idx]:
            self.last_time_query_idx += 1
        self.last_time_query_idx = min(self.last_time_query_idx, len(self.times)-1)

        state = self.states[self.last_time_query_idx]
        self.goal = state[2:4]
        return self.states

def get_entire_distance(positions: np.ndarray):
    distance = np.zeros(positions.shape[0])

    for i in range(1, positions.shape[0]):
        distance[i] = np.linalg.norm(positions[i] - positions[i-1])

    return np.sum(distance)

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
