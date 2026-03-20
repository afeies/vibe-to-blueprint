import numpy as np
import cv2
from PIL import Image
from shapely.geometry import box

def make_edge_map(rooms: list, size=512) -> Image.Image:
    img = np.zeros((size, size), dtype=np.uint8)
    n = len(rooms)
    w = size // max(n, 1)
    for i in range(n):
        x0, y0 = i * w + 10, 60
        x1, y1 = (i + 1) * w - 10, size - 60
        cv2.rectangle(img, (x0, y0), (x1, y1), 255, 2)
        # add a window rectangle
        wx = (x0 + x1) // 2 - 30
        cv2.rectangle(img, (wx, y0 + 20), (wx + 60, y0 + 60), 180, 1)
    edges = cv2.Canny(img, 50, 150)
    return Image.fromarray(edges).convert("RGB")