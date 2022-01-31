from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from environment import road, graph
from path.utils import pixel_to_xy, get_rectangle_corners
from path.carVisualizer import CarVisualizer
from performance.cache_utils import cached
from physics.collisions import is_colliding

# The MapVisualizer object deals with the intricacies of displaying the map in the plots.
class MapVisualizer:
    def __init__(self, road_constants: dict) -> None:
        """ Initialize the visualizer with all the input parameters.
        """
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']

        road_map, road_graph = road.get_road_info((self.lat, self.lon), self.zoom, max_regularization_dist=self.regularization, res_zoom_upsample=self.upsampling)
        
        self.road_edges: np.ndarray = road.get_edge_img(road_map).T
        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph

        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)
        xm, xM = 0, self.meters_per_pixel*self.map.shape[1]
        ym, yM = 0, self.meters_per_pixel*self.map.shape[0]
        sideX = xM - xm
        sideY = yM - ym
        self.xm = xm - sideX/2 + self.meters_per_pixel
        self.xM = xM - sideX/2 + self.meters_per_pixel
        self.ym = ym - sideY/2
        self.yM = yM - sideY/2
        self.extent = [self.xm, self.xM, self.ym, self.yM]

        self.block_positions, self.block_edges = self.load_road_blocks(self.road_edges, self.meters_per_pixel)

        self.is_colliding = False
        self.collision_count = 0
    
    @cached(class_func=True, folder="road blocks")
    def load_road_blocks(self, road_edges: np.ndarray, meters_per_pixel: float, block_size_multiplier: float = 0.999) -> np.ndarray:
        """ Loads the road blocks from the edges. Used for collisions.
        """
        block_pixels_x, block_pixels_y = np.where(road_edges > 0)
        block_pixels = np.stack((block_pixels_x, block_pixels_y), axis=1)
        block_positions = np.array([pixel_to_xy(block, self.map.shape, self.meters_per_pixel) for block in block_pixels])
        block_side = block_size_multiplier*meters_per_pixel
        block_edges = np.array([get_rectangle_corners(block, block_side, block_side) for block in block_positions])
        return block_positions, block_edges

    def collision_counter(self, car_repr: CarVisualizer, square_window_side: int = 5, visualization: bool = False) -> int:
        """ Keeps track of the number of collisions along a simulation.
        """
        self.is_currently_colliding = self.car_collides_with_road(car_repr, square_window_side, visualization=visualization)
        if self.is_currently_colliding and not self.is_colliding:
            self.collision_count += 1
        self.is_colliding = self.is_currently_colliding
        return self.collision_count


    def car_collides_with_road(self, car_repr: CarVisualizer, square_window_side: int = 5, visualization: bool = False) -> bool:
        """ Determines whether the car is colliding with the road edges, using the separating axis theorem.
        """
        _, _, x, y, _ = car_repr.state
        admissible_block_mask_x = np.logical_and(
            self.block_positions[:, 0] > x - square_window_side, 
            self.block_positions[:, 0] < x + square_window_side
        )
        admissible_block_mask_y = np.logical_and(
            self.block_positions[:, 1] > y - square_window_side,
            self.block_positions[:, 1] < y + square_window_side
        )
        admissible_block_mask = np.logical_and(admissible_block_mask_x, admissible_block_mask_y)
        admissible_blocks = self.block_edges[admissible_block_mask]

        car_rects = car_repr.get_car_representation(car_repr.state)
        car_parts = np.array_split(car_rects, 4, 0)
        for car_part in car_parts:
            if is_colliding(car_part, *admissible_blocks):
                if visualization and not self.is_colliding:
                    ax = plt.gca()
                    ax.add_patch(plt.Polygon(car_part, closed=True, fill=False, edgecolor='red'))
                    for obj3 in admissible_blocks:
                        ax.add_patch(plt.Polygon(obj3, closed=True, fill=False, edgecolor='blue'))
                return True
        return False

    def plot(self, ax: plt.Axes):
        """ Plots the map.
        """
        ax.imshow(self.map, extent=self.extent, cmap='gray')

