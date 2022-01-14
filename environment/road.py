from typing import Union
from .graph import WeightedGraph, get_graph_from_binary_image, draw_graph, draw_path
import cv2
import numpy as np
import requests
import os

def retrieve_api_key(key_file_name: str = "api.key", search_pardir = True) -> str:
    """Retrieve Google Maps API key from file.
    Returns:
        Google Maps API key.
    """
    if not os.path.exists(key_file_name):
        if search_pardir:
            key_file_name = os.path.join(os.pardir, key_file_name)
            if not os.path.exists(key_file_name):
                raise FileNotFoundError(f"File {key_file_name} not found.")
        else:
            raise FileNotFoundError(f"File {key_file_name} not found.")
    with open(key_file_name, "r") as f:
        api_key = f.read().strip()

    return api_key


def zoom_to_scale(zoom: int, latitude: float):
    """
    Calculate the scale of the map at a given zoom level and latitude.
    Source: https://groups.google.com/g/google-maps-js-api-v3/c/hDRO4oHVSeM/m/osOYQYXg2oUJ

    Args:
        zoom: Zoom level.
        latitude: Latitude of the center of the map.
    Returns:
        Scale of the map.
    """
    meters_per_pixel = 156543.03392 * np.cos(latitude * np.pi / 180) / np.power(2, zoom) 
    return meters_per_pixel

def get_road_image(center: tuple[float, float], zoom: int, 
                    size: tuple[int, int] = (400, 400), 
                    style_map_id: str = "6e80ae00ec0ca703", img_caching = True, 
                    single_channel: bool= False, api_key: Union[None, str] = None) -> np.ndarray:
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
    if api_key is None:
        api_key = retrieve_api_key()
    img_cache_dir = "cache"
    # bottom 20 pixels contain watermark, therefore we increase the size and crop
    # the bottom 20 pixels - the center must, then, be adjusted accordingly
    watermark_size = 20
    earth_radius_m = 6378137

    # scale (mercator projection) 
    meters_per_pixel = zoom_to_scale(zoom, center[0])

    # adjust center
    meter_center_adjustment = watermark_size * meters_per_pixel / 2
    degree_center_adjustment = np.arcsin(meter_center_adjustment / earth_radius_m) * 180 / np.pi
    center = (center[0] - degree_center_adjustment, center[1])

    # build url
    url = "https://maps.googleapis.com/maps/api/staticmap?" \
            f"center={center[0]},{center[1]}&size={size[0]}x{size[1] + watermark_size}" \
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

    if single_channel:
        img = np.sum(img, 2)
        np.where(img > 255, 255, img)
        img = img.astype(np.uint8)
    return img

def get_road_info(*args, max_regularization_dist = np.inf, **kwargs) -> tuple[np.ndarray, WeightedGraph]:
    """Get road image from Google Maps API and build road graph.
    Check documentation for get_road_image for arguments.

    Args:
        *args: Arguments for get_road_image.
        **kwargs: Keyword arguments for get_road_image.
    """
    img = get_road_image(*args, **kwargs)
    road_graph = get_graph_from_binary_image((np.sum(img, 2) if len(img.shape) > 2 else img) > 0.5, max_regularization_dist = max_regularization_dist)
    return img, road_graph


if __name__ == '__main__':
    #img, graph = get_road_image((38.72, -9.15), 15) #random lisbon place
    img, graph = get_road_info((38.7367256,-9.1388871), 16, max_regularization_dist=20) # ist
    
    img_graph = draw_graph(img, graph, transpose=True)
    
    # example demonstrating shortest path calculation
    source, end = list(graph.connections.keys())[1], list(graph.connections.keys())[250]
    parents, distances = graph.get_mst_dijkstra(source)
    path = graph.get_path_from_mst(end, parents, distances)

    #print(graph)
    print(f"Shortest path from {source} to {end}: {path}")
    print(f"Shortest path length in nodes: {len(path)}")
    print(f"Shortest path length in meters: {graph.get_path_cost(path)}")

    img_path = draw_path(img_graph, path, edge_color = (255, 255, 255), node_color = (255, 255, 255), transpose=True)

    cv2.imshow('img_path', img_path)
    cv2.imshow('img_graph', img_graph)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

