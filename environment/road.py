import cv2
import numpy as np
import requests

def get_road_image(api_key: str, center: tuple[float, float], zoom: int, 
                    size: tuple[int, int] = (400, 400), 
                    style_map_id: str = "6e80ae00ec0ca703") -> np.ndarray:
    """
    Get road image from Google Maps API.
    Args:
        api_key: Google Maps API key.
        center: Center of the map. Latitude and longitude.
        zoom: Zoom level.
        size: Size of the map in pixels.
        style_map_id: Style map id, used to apply styling.
    Returns:
        Road image.
    """
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

    # get image from request and convert to cv2 image
    response = requests.get(url)
    array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    return img[:-watermark_size, :, :]


if __name__ == '__main__':
    import os
    api_key = open(os.path.join(os.pardir, "api.key")).read().strip()
    img = get_road_image(api_key, (38.72, -9.15), 15)
    
    # cv2 imshow
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
