from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_rectangle_corners, figure_number
from environment import road, graph


class MapVisualizer:
    def __init__(self, road_constants: dict, interp_method: Union[str, None] = 'none') -> None:
        self.lat = road_constants['lat']
        self.lon = road_constants['lon']
        self.zoom = road_constants['zoom']
        self.upsampling = road_constants['upsampling']
        self.regularization = road_constants['regularization']

        road_map, road_graph = road.get_road_info((self.lat, self.lon), self.zoom, max_regularization_dist=self.regularization, res_zoom_upsample=self.upsampling)
        
        self.map: np.ndarray = road_map
        self.graph: graph.WeightedGraph = road_graph

        self.meters_per_pixel = road.zoom_to_scale(self.zoom + self.upsampling, self.lat)
        xm, xM = 0, self.meters_per_pixel*self.map.shape[1]
        ym, yM = 0, self.meters_per_pixel*self.map.shape[0]
        sideX = xM - xm
        sideY = yM - ym
        self.xm = xm - sideX/2
        self.xM = xM - sideX/2
        self.ym = ym - sideY/2
        self.yM = yM - sideY/2
        self.extent = [self.xm, self.xM, self.ym, self.yM]

        self.interp_method = interp_method
    
    def plot(self, state, window: tuple = ((-20, 20), (-20, 20)), block: bool = False, clf: bool = False) -> None:
        plt.figure(figure_number)
        if clf: plt.clf()
        x, y = state[0], state[1]

        plt.imshow(self.map, interpolation=self.interp_method, extent=self.extent, cmap='gray')

        xlim = (window[0][0] + x, window[0][1] + x)
        ylim = (window[1][0] + y, window[1][1] + y)
        plt.xlim(xlim)
        plt.ylim(ylim)     
        plt.show(block=block)