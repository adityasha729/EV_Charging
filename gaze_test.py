import numpy as np

def get_charging_page_sections():
    # Define rectangles as (x1_pct, y1_pct, x2_pct, y2_pct)
    return {
        # "header": (0.0, 0.00, 1.0, 0.10),
        # "charging_icon": (0.35, 0.12, 0.65, 0.24),
        # "countdown": (0.25, 0.24, 0.75, 0.33),
        
        "kiosk_1": (0, 0, 0.23, 1.0),
        "kiosk_2": (0.23, 0.0, 0.46, 1.0),
        "kiosk_3": (0.46, 0.0, 0.69, 1.0),
        "battery_container": (0.69, 0.0, 1.0, 1.0),
        # "eco_message": (0.18, 0.72, 0.82, 0.82),
        # "footer": (0.0, 0.88, 1.0, 1.0),
        # "unknown": (0,0,0,0)
    }

def point_in_rect(x, y, rect, width=1920, height=1080):
    x1_pct, y1_pct, x2_pct, y2_pct = rect
    x1, y1, x2, y2 = x1_pct * width, y1_pct * height, x2_pct * width, y2_pct * height
    return x >= x1 and x <= x2 and y >= y1 and y <= y2

def map_gaze_to_section(x, y, screen_width=1920, screen_height=1080):
    sections = get_charging_page_sections()
    for name, rect in sections.items():
        if name == "unknown":
            continue
        if point_in_rect(x, y, rect, screen_width, screen_height):
            return name
    return "unknown"

def map_numpy_gaze_array(coords, screen_w=1920, screen_h=1080, detect_normalized=True):
    """
    Map an array-like of gaze points to section names.
    coords: array-like shape (N,2) with columns [x, y].
      - If values look normalized (max <= 1.0) and detect_normalized is True,
        they will be scaled to pixels using screen_w/screen_h.
      - Otherwise treated as pixel coordinates.
    Returns: list of section names (length N).
    """
    
    a = np.asarray(coords)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("coords must be shape (N,2)")
    xs = a[:, 0].astype(float)
    ys = a[:, 1].astype(float)
    if detect_normalized:
        try:
            if np.nanmax(xs) <= 1.0 and np.nanmax(ys) <= 1.0:
                xs = xs * screen_w
                ys = ys * screen_h
        except Exception:
            pass
    sections = [map_gaze_to_section(float(x), float(y), screen_w, screen_h) for x, y in zip(xs, ys)]
    return sections

print("Gaze mapping module loaded.")
print(map_numpy_gaze_array([[1900,1000],[0.5,0.5],[960,540]]))