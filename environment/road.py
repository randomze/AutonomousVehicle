from typing import TypedDict
import cv2
import numpy as np
import requests
import os
import random

class WeightedGraph:
    def __init__(self):
        self.connections = {}
    def add_connection(self, node1, node2, weight=1):
        if node1 == node2:
            raise ValueError('Node1 cannot be the same as node2')
        if node1 not in self.connections:
            self.connections[node1] = {}
        self.connections[node1][node2] = weight
        if node2 not in self.connections:
            self.connections[node2] = {}
        self.connections[node2][node1] = weight
    def remove_connection(self, node1, node2):
        if node1 not in self.connections:
            return
        if node2 not in self.connections[node1]:
            return
        del self.connections[node1][node2]
        del self.connections[node2][node1]
    def remove_node(self, node):
        if node not in self.connections:
            return
        for node2 in self.connections[node]:
            self.remove_connection(node, node2)
        del self.connections[node]
    def get_connections(self, node):
        if node not in self.connections:
            return []
        return self.connections[node].keys()
    def is_connected(self, node1, node2):
        if node1 not in self.connections:
            return False
        if node2 not in self.connections[node1]:
            return False
        return True
    def get_all_nodes(self):
        return self.connections.keys()
    def get_weight(self, node1, node2):
        if node1 not in self.connections:
            return None
        if node2 not in self.connections[node1]:
            return None
        return self.connections[node1][node2]
    def get_path_cost(self, path): # a path is a list of nodes
        cost = 0
        for i in range(len(path)-1):
            cost += self.get_weight(path[i], path[i+1])
        return cost
    def replace_node_with_edge(self, middle_node):
        node1, node2 = self.connections[middle_node]
        self.add_connection(node1, node2, self.connections[middle_node][node1] + self.connections[middle_node][node2])
        self.remove_connection(node1, middle_node)
        self.remove_connection(node2, middle_node)
    def __str__(self) -> str:
        return str(self.connections)
        

def get_road_image(api_key: str, center: tuple[float, float], zoom: int, 
                    size: tuple[int, int] = (400, 400), 
                    style_map_id: str = "6e80ae00ec0ca703", img_caching = True) -> np.ndarray:
    """
    Get road image from Google Maps API.
    Args:
        api_key: Google Maps API key.
        center: Center of the map. Latitude and longitude.
        zoom: Zoom level.
        size: Size of the map in pixels.
        style_map_id: Style map id, used to apply styling.
        img_caching: Use caching to minimize number of API calls.
    Returns:
        Road image.
    """
    img_cache_dir = "cache"
    # bottom 20 pixels contain watermark, therefore we increase the size and crop
    # the bottom 20 pixels - the center must, then, be adjusted accordingly
    watermark_size = 20
    earth_radius_m = 6378137

    # scale (mercator projection) https://groups.google.com/g/google-maps-js-api-v3/c/hDRO4oHVSeM/m/osOYQYXg2oUJ
    meters_per_pixel = 156543.03392 * np.cos(center[0] * np.pi / 180) / np.power(2, zoom) 

    # adjust center
    meter_center_adjustment = watermark_size * meters_per_pixel / 2
    degree_center_adjustment = np.arcsin(meter_center_adjustment / earth_radius_m) * 180 / np.pi
    center = (center[0] - degree_center_adjustment, center[1])

    # build url
    url = "https://maps.googleapis.com/maps/api/staticmap?" \
            f"center={center[0]},{center[1]}&size={size[0]}x{size[1]}" \
            f"&zoom={zoom}&map_id={style_map_id}&key={api_key}"

    if img_caching:
        # cache image
        img_name = f"c={center[0]},{center[1]};s={size[0]}x{size[1]};z={zoom};id={style_map_id}"
        directory = os.path.join(os.path.dirname(__file__), img_cache_dir)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        img_path = os.path.join(directory, img_name + ".png")
        if not os.path.isfile(img_path):
            print("Cached image not found, downloading...")
            response = requests.get(url)
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            img = img[:-watermark_size, :, :]
            cv2.imwrite(img_path, img)
        img = cv2.imread(img_path)
    else:
        # get image from url
        response = requests.get(url)
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
        # crop bottom 20 pixels
        img = img[:-watermark_size, :, :]
    return img

def get_road_graph(img_binary: np.ndarray, max_regulatization_iter: int = 5) -> np.ndarray:
    """
    Get road graph from road image.
    Args:
        img: Road image, thinned.
        threshold: Threshold for road pixels.
    Returns:
        Weighted road graph. Weighted by distance.
    """
    road_graph = WeightedGraph()
    for i in range(1, img_binary.shape[0]-1):
        for j in range(1, img_binary.shape[1]-1):
            if not img_binary[i, j]: continue
            if img_binary[i-1, j]: 
                road_graph.add_connection((i, j), (i-1, j), 1)
            if img_binary[i+1, j]: 
                road_graph.add_connection((i, j), (i+1, j), 1)
            if img_binary[i, j-1]: 
                road_graph.add_connection((i, j), (i, j-1), 1)
            if img_binary[i, j+1]: 
                road_graph.add_connection((i, j), (i, j+1), 1)
            con_north = road_graph.is_connected((i, j), (i-1, j))
            con_south = road_graph.is_connected((i, j), (i+1, j))
            con_west = road_graph.is_connected((i, j), (i, j-1))
            con_east = road_graph.is_connected((i, j), (i, j+1))
            if not con_north and not con_west:
                if img_binary[i-1, j-1]: 
                    road_graph.add_connection((i, j), (i-1, j-1), np.sqrt(2))
            if not con_north and not con_east:
                if img_binary[i-1, j+1]: 
                    road_graph.add_connection((i, j), (i-1, j+1), np.sqrt(2))
            if not con_south and not con_west:
                if img_binary[i+1, j-1]: 
                    road_graph.add_connection((i, j), (i+1, j-1), np.sqrt(2))
            if not con_south and not con_east:
                if img_binary[i+1, j+1]: 
                    road_graph.add_connection((i, j), (i+1, j+1), np.sqrt(2))
    print(f"Regularizing graph with {len(road_graph.connections)}...")
    def regularize(graph: WeightedGraph) -> int:
        iterations = 0
        nodes_deleted = 0
        nodes_deleted_iter = 1
        while nodes_deleted_iter > 0 and iterations < max_regulatization_iter:
            nodes_deleted_iter = 0
            for connection in graph.connections:
                if len(graph.connections[connection]) == 2: # just remove node
                    graph.replace_node_with_edge(connection)
                    nodes_deleted_iter += 1
            nodes_deleted += nodes_deleted_iter
            for connection in list(graph.connections):
                if graph.connections[connection] == {}:
                    graph.remove_node(connection)
            iterations += 1
        return nodes_deleted
    deleted = regularize(road_graph)
    print(f"Regularization deleted {deleted} nodes.")
    return road_graph

if __name__ == '__main__':
    key_fname = 'api.key'
    if not os.path.exists(key_fname):
        key_fname = os.path.join(os.pardir, key_fname)

    api_key = open(key_fname).read().strip()
    #img_color = get_road_image(api_key, (38.72, -9.15), 15) #random lisbon place
    img_color = get_road_image(api_key, (38.7367256,-9.1388871), 16) # ist
    
    img_binary = np.sum(img_color, 2) > 0.5
    img_gray = np.zeros_like(img_color)
    for i in range(3):
        img_gray[:, :, i] = img_binary*255
    img_color = img_gray
    L = img_color.shape[0]

    img1channel = img_binary.astype(np.uint8)*255

    thinned = cv2.ximgproc.thinning(img1channel)

    thinned_binary = thinned.astype(np.uint8) > 0
    road_graph = get_road_graph(thinned_binary)
    #convert thinned to color
    thinned_color = np.zeros_like(img_color)
    for i in range(3):
        thinned_color[:, :, i] = thinned_binary*255
    img_graph = thinned_color.copy()
    # draw circles at nodes and lines at edges
    inv = lambda pos: (pos[1], pos[0])
    for node in road_graph.connections:
        cv2.circle(img_graph, inv(node), 2, (0, 0, 255), 2)
        for edge in road_graph.connections[node]:
            cv2.line(img_graph, inv(node), inv(edge), (0, 255, 0), 1)


    #print(road_graph)

    nlines = 50
    thresh = 10
    rho_res = 1
    theta_res = np.pi/180
    
    lines = cv2.HoughLines(thinned,rho_res,theta_res,thresh)
    random.shuffle(lines)
    img_color_cpy = img_color.copy()
    for line in lines[:nlines]:
        for rho,theta in line:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + (L/2)*(-b))
            y1 = int(y0 + (L/2)*(a))
            x2 = int(x0 - (L/2)*(-b))
            y2 = int(y0 - (L/2)*(a))

            p1 = (rho/np.cos(theta), 0)
            p2 = (rho/np.cos(theta), L)

            cv2.line(img_color_cpy,(x1,y1),(x2,y2),(255,255,0),2)
    
    img_P = img_color.copy()
    linesP = cv2.HoughLinesP(thinned,rho_res,theta_res,thresh,minLineLength=10,maxLineGap=5)
    for x1,y1,x2,y2 in linesP[0]:
        cv2.line(img_P,(x1,y1),(x2,y2),(255,255,0),2)

    cv2.imshow('img_graph', img_graph)
    cv2.imshow('original', img_color)
    cv2.imshow('thinned', thinned)
    cv2.imshow('Hough Lines', img_color_cpy)
    cv2.imshow('Hough Lines P', img_P)

    cv2.waitKey(0)
    cv2.destroyAllWindows()