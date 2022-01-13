import numpy as np
import cv2

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
    def get_shortest_path_dijkstra(self, start_node, end_node):
        if start_node == end_node:
            return [start_node]

        if start_node not in self.connections:
            raise ValueError('Start node not in graph')
        if end_node not in self.connections:
            raise ValueError('End node not in graph')
        
        unvisited = set(self.get_all_nodes())
        distances = {node: float('inf') for node in unvisited}
        distances[start_node] = 0
        previous = {node: None for node in unvisited}
        
        while unvisited:
            current_node = min(unvisited, key=lambda node: distances[node])
            unvisited.remove(current_node)
            if current_node == end_node:
                break
            for neighbor in self.get_connections(current_node):
                if neighbor in unvisited:
                    neighbor_distance = distances[current_node] + self.get_weight(current_node, neighbor)
                    if neighbor_distance < distances[neighbor]:
                        distances[neighbor] = neighbor_distance
                        previous[neighbor] = current_node
        
        path = [end_node]
        while previous[end_node] is not None:
            path.append(previous[end_node])
            end_node = previous[end_node]
        path.reverse()
        return path
    def __str__(self) -> str:
        return str(self.connections)
    

def get_graph_from_binary_image(img_binary: np.ndarray, max_regulatization_iter: int = 5, max_regularization_dist: float = 20.0, verbose: bool = False) -> np.ndarray:
    """
    Get road graph from road image. Image must be binary.
    Args:
        img_binary: Binary image to get graph from.
        max_regulatization_iter: Maximum number of iterations to perform regularization.
                                (generally 99% of regularization is performed on 1 iteration)
        max_regularization_dist: Maximum distance to perform regularization.
        verbose: Verbosity level.
    Returns:
        Weighted graph - weighted by distance.
    """
    
    img1channel = img_binary.astype(np.uint8)*255
    thinned = cv2.ximgproc.thinning(img1channel)
    thinned_binary = thinned.astype(np.uint8) > 0
    graph = WeightedGraph()
    for i in range(1, thinned_binary.shape[0]-1):
        for j in range(1, thinned_binary.shape[1]-1):
            if not thinned_binary[i, j]: continue
            if thinned_binary[i-1, j]: 
                graph.add_connection((i, j), (i-1, j), 1)
            if thinned_binary[i+1, j]: 
                graph.add_connection((i, j), (i+1, j), 1)
            if thinned_binary[i, j-1]: 
                graph.add_connection((i, j), (i, j-1), 1)
            if thinned_binary[i, j+1]: 
                graph.add_connection((i, j), (i, j+1), 1)
            con_north = graph.is_connected((i, j), (i-1, j))
            con_south = graph.is_connected((i, j), (i+1, j))
            con_west = graph.is_connected((i, j), (i, j-1))
            con_east = graph.is_connected((i, j), (i, j+1))
            if not con_north and not con_west:
                if thinned_binary[i-1, j-1]: 
                    graph.add_connection((i, j), (i-1, j-1), np.sqrt(2))
            if not con_north and not con_east:
                if thinned_binary[i-1, j+1]: 
                    graph.add_connection((i, j), (i-1, j+1), np.sqrt(2))
            if not con_south and not con_west:
                if thinned_binary[i+1, j-1]: 
                    graph.add_connection((i, j), (i+1, j-1), np.sqrt(2))
            if not con_south and not con_east:
                if thinned_binary[i+1, j+1]: 
                    graph.add_connection((i, j), (i+1, j+1), np.sqrt(2))
    if verbose: print(f"Regularizing graph with {len(graph.connections)}...")
    def regularize(graph: WeightedGraph) -> int:
        iterations = 0
        nodes_deleted = 0
        nodes_deleted_iter = 1
        while nodes_deleted_iter > 0 and iterations < max_regulatization_iter:
            nodes_deleted_iter = 0
            for connection in graph.connections:
                if len(graph.connections[connection]) == 2: # just remove node
                    node1, node2 = graph.connections[connection].keys()
                    if graph.connections[connection][node1] > max_regularization_dist or graph.connections[connection][node2] > max_regularization_dist:
                        continue
                    graph.replace_node_with_edge(connection)
                    nodes_deleted_iter += 1
            nodes_deleted += nodes_deleted_iter
            for connection in list(graph.connections):
                if graph.connections[connection] == {}:
                    graph.remove_node(connection)
            iterations += 1
        return nodes_deleted
    deleted = regularize(graph)
    if verbose: print(f"Regularization deleted {deleted} nodes, leaving {len(graph.connections)} nodes.")
    return graph

def draw_graph(img: np.ndarray, graph: WeightedGraph, transpose: bool = False,
                edge_color: tuple[int, int, int] = (255, 0, 255),
                node_color: tuple[int, int, int] = (0, 0, 255),
                edge_thickness: int = 1, node_thickness: int = 2,
                node_radius: int = 2 ) -> np.ndarray:
    """
    Draw graph on image. Nodes must be (x, y) tuples.
    Args:
        img: 3 channel image to draw graph on. 
        graph: Graph to draw.
    Returns:
        Image with graph drawn.
    """
    pos = lambda p: (p[1], p[0]) if transpose else p
    img_copy = img.copy()
    for node in graph.connections:
        cv2.circle(img_copy, pos(node), node_radius, node_color, node_thickness)
        for edge in graph.connections[node]:
            cv2.line(img_copy, pos(node), pos(edge), edge_color, edge_thickness)
    return img_copy

