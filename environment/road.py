from enum import Enum
from typing import Union
from . import graph
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

def upsampling_centers(initial_center, size, zoom, upsampling):
    """Upsample the center of the road image.
    Args:
        initial_center: Center of the road image.
        zoom: Zoom level.
        upsampling: Upsampling factor.
    Returns:
        List of centers.
    """
    ic_rad = (initial_center[0]*np.pi/180, initial_center[1]*np.pi/180)
    earth_radius = 6378137
    zoom = zoom + upsampling
    meter_per_pixel = zoom_to_scale(zoom, initial_center[0])
    meter_height = size[0] * meter_per_pixel
    meter_width = size[1] * meter_per_pixel
    lat_range = np.arcsin((meter_height/2) / earth_radius) 
    lon_range = np.arcsin((meter_width/2) / earth_radius) / np.cos(initial_center[0] * np.pi / 180) # adjust for latitude (mercator projection)
    centers_rad = []
    centers_rad.append((ic_rad[0] + lat_range, ic_rad[1] - lon_range))
    centers_rad.append((ic_rad[0] + lat_range, ic_rad[1] + lon_range))
    centers_rad.append((ic_rad[0] - lat_range, ic_rad[1] - lon_range))
    centers_rad.append((ic_rad[0] - lat_range, ic_rad[1] + lon_range))
    center_deg = [(c[0]*180/np.pi, c[1]*180/np.pi) for c in centers_rad]
    return center_deg


def get_road_image(center: tuple[float, float], zoom: int,
                    res_zoom_upsample: int = 0, margin_px: int = 10) -> np.ndarray:
    """Stitch several images together to create a single, upsampled, image
    Zoom + res_zoom_upsample is the zoom level of the resulting image.
    Road pixel size stays the same for zoom + res_zoom_upsample > 19, which can
    be problematic for certain applications. 
    zoom + res_zoom_upsample == 20 has the least noticeable impact.

    Args:
        center: Center of the road image - latitude and longitude.
        zoom: Zoom level (google maps standard).
        res_zoom_upsample: Resolution zoom level (upsampling). 
        margin_px: Margin of each border in pixels. Useful for applying 
                morphological operations.
    Returns:
        Road image.
    """
    size_img_standard = (400, 400)
    size_w_margin = (size_img_standard[0] + 2*margin_px, size_img_standard[1] + 2*margin_px)
    if res_zoom_upsample == 0:
        return get_single_road_image(center, zoom, size_w_margin)
    size_per_img = (400 * (2 ** (res_zoom_upsample-1)), 400 * (2 ** (res_zoom_upsample-1)))
    centers = upsampling_centers(center, size_per_img, zoom, res_zoom_upsample)
    res_x = size_per_img[0]*2 + 2*margin_px
    res_y = size_per_img[1]*2 + 2*margin_px
    big_img = np.zeros((res_y, res_x), dtype=np.uint8)
    topleft_img = get_road_image(centers[0], zoom+1, res_zoom_upsample-1, margin_px)
    topright_img = get_road_image(centers[1], zoom+1, res_zoom_upsample-1, margin_px)
    bottomleft_img = get_road_image(centers[2], zoom+1, res_zoom_upsample-1, margin_px)
    bottomright_img = get_road_image(centers[3], zoom+1, res_zoom_upsample-1, margin_px)
    big_img[0:size_per_img[0] + 2*margin_px, 0:size_per_img[1] + 2*margin_px] = topleft_img
    big_img[0:size_per_img[0] + 2*margin_px, size_per_img[1]:size_per_img[1]*2 + 2*margin_px] = topright_img
    big_img[size_per_img[0]:size_per_img[0]*2 + 2*margin_px, 0:size_per_img[1] + 2*margin_px] = bottomleft_img
    big_img[size_per_img[0]:size_per_img[0]*2 + 2*margin_px, size_per_img[1]:size_per_img[1]*2 + 2*margin_px] = bottomright_img
    return big_img


def get_single_road_image(center: tuple[float, float], zoom: int, 
                    size: tuple[int, int] = (400, 400)) -> np.ndarray:
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
    api_key = retrieve_api_key()
    img_cache_dir = "cache"
    threshold_valid_pixel = 30
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
            f"&zoom={zoom}&map_id=6e80ae00ec0ca703&key={api_key}"

    # cache image
    img_name = f"c={center[0]},{center[1]};s={size[0]}x{size[1]};z={zoom}"
    directory = os.path.join(os.path.dirname(__file__), img_cache_dir)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    img_path = os.path.join(directory, img_name + ".png")
    if not os.path.isfile(img_path):
        print("Cached image not found, downloading...")
        try:
            response = requests.get(url)
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            img = img[:-watermark_size, :, :]
        except TypeError as exc:
            print(f"{response.content = }")
            print(f"{center = }")
            print(f"{size = }")
            print(f"{zoom = }")
            raise exc
        cv2.imwrite(img_path, img)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    img = np.sum(img, 2)
    np.where(img > threshold_valid_pixel, 255, img)
    img = img.astype(np.uint8)
    return img

class BorderType(Enum):
    """
    Types of borders.
    """
    NO_OP = 0
    FULL_0 = 1
    FULL_1 = 2

def get_edge_img(binary_img: np.ndarray, border_type: BorderType = BorderType.FULL_1) -> np.ndarray:
    """Get road edge image from road image. Useful for road edge detection.
    Uses the hit-or-miss transform. 

    Args:
        img: Road image.
        border_type: Type of border. If a border is selected, a larger image is created.
    Returns:
        Road edge image.
    """
    # invert image
    binary_img = 255 - binary_img

    structure_element_8_connected = np.array([
        [1, 1, 1], 
        [1, 1, 1], 
        [1, 1, 1]], dtype=np.uint8)

    # hit-or-miss transform
    img_edge_4_connected = cv2.morphologyEx(binary_img, cv2.MORPH_HITMISS, structure_element_8_connected)
    ret_img = np.abs(img_edge_4_connected - binary_img)

    if border_type == BorderType.FULL_0:
        ret_img = cv2.copyMakeBorder(ret_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    elif border_type == BorderType.FULL_1:
        ret_img = cv2.copyMakeBorder(ret_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    return ret_img

def get_road_info(*args, max_regularization_dist = np.inf, **kwargs) -> tuple[np.ndarray, graph.WeightedGraph]:
    """Get road image from Google Maps API and build road graph.
    Check documentation for get_road_image for arguments.

    Args:
        *args: Arguments for get_road_image.
        **kwargs: Keyword arguments for get_road_image.
    """
    img = get_road_image(*args, **kwargs)
    road_graph = graph.get_graph_from_binary_image((np.sum(img, 2) if len(img.shape) > 2 else img) > 0.5, max_regularization_dist = max_regularization_dist)
    return img, road_graph

# btw, run this script from base directory with python -m environment.road
if __name__ == '__main__':
    #img, graph = get_road_image((38.72, -9.15), 15) #random lisbon place
    img, r_graph = get_road_info((38.7367256,-9.1388871), 16, max_regularization_dist=200) # ist
    # convert to 3 channel image
    edges = get_edge_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img_graph = graph.draw_graph(img, r_graph, transpose=True)
    
    # example demonstrating shortest path calculation
    source, end = list(r_graph.connections.keys())[1], list(r_graph.connections.keys())[50]
    parents, distances = r_graph.get_mst_dijkstra(source)
    path = r_graph.get_path_from_mst(end, parents, distances)

    #print(graph)
    print(f"Shortest path from {source} to {end}: {path}")
    print(f"Shortest path length in nodes: {len(path)}")
    print(f"Shortest path length in meters: {r_graph.get_path_cost(path)}")

    img_path = graph.draw_path(img_graph, path, edge_color = (0, 255, 255), node_color = (0, 255, 255), transpose=True)

    cv2.imshow("edges", edges)
    cv2.imshow('img_path', img_path)
    cv2.imshow('img_graph', img_graph)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

