import numpy as np
import matplotlib.pyplot as plt
from environment import road, graph
from visualization.utils import pixel_to_xy, xy_to_pixel, figure_number

class TrajectoryGenerator:

    def __init__(self, road_constants: dict, path: tuple):
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']

        road_map, road_graph = road.get_road_info((self.lat, self.lon), self.zoom, max_regularization_dist=self.regularization, res_zoom_upsample=self.upsampling)

        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph
        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)
        
        self.path = self.get_xy_path(*path)
        self.goal = None


    def get_xy_path(self, init_point, final_point):
        nodes = list(self.graph.get_all_nodes())
        # invert x and y due to graph implementation
        init_point = (init_point[1], init_point[0])
        final_point = (final_point[1], final_point[0])

        # get first node as the closest to init_point
        init_pixel = xy_to_pixel(init_point[0], init_point[1], self.map.shape, self.meters_per_pixel)
        init_node = nodes[np.argmin([np.linalg.norm(np.array(init_pixel) - np.array(node)) for node in nodes])]

        # get last node as the closest to final_point
        final_pixel = xy_to_pixel(final_point[0], final_point[1], self.map.shape, self.meters_per_pixel)
        final_node = nodes[np.argmin([np.linalg.norm(np.array(final_pixel) - np.array(node)) for node in nodes])]

        # get the path from init_node to final_node
        mst = self.graph.get_mst_dijkstra(init_node)
        nodes_to_traverse = self.graph.get_path_from_mst(final_node, *mst)
        nodes_to_traverse = [(node[0], node[1]) for node in nodes_to_traverse] # not sure why this is needed, but it works
        points_to_traverse = [pixel_to_xy(node, self.map.shape, self.meters_per_pixel) for node in nodes_to_traverse]
        points_to_traverse.insert(0, init_point)
        points_to_traverse.append(final_point)

        # again, invert x, y definition due to graph implementation
        points_to_traverse = [(point[1], point[0]) for point in points_to_traverse]
        return points_to_traverse

    def cur_goal_xy(self, instant: float, step: float = 20.0):
        path_progress_index = int(instant / step)
        path_progress_index = min(path_progress_index, len(self.path)-1)
        self.goal = self.path[path_progress_index]
        return self.goal

    def plot(self, clf: bool = False, block: bool = False, color ='r', cur_color = 'b'):
        plt.figure(figure_number)
        if clf: plt.clf()
        for point in self.path:
            plt.scatter(point[0], point[1], color=color)

        if self.goal is not None:
            plt.scatter(self.goal[0], self.goal[1], color=cur_color)
        plt.show(block=block)

    def output(self, instant):
        pos = self.cur_goal_xy(instant)
        return np.array([0.5, np.pi/2, pos[0], pos[1], 0.0])
