import random
import cv2
import sys
from pathlib import Path
from loguru import logger
import numpy as np

VID_FORMATS = ["asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"]

def random_bgr():
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)
    return (blue, green, red)

def convert_bbox_to_z(bbox):
    """
    [x1,y1,x2,y2] -> [x,y,s,r]
    (x1,y1) -> left top, (x2,y2) -> right bottom
    x,y = center of box coordinate, s = scale/area, r = aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
SCRIPT = ROOT / "script"
REQUIREMENTS = ROOT / "requirements.txt"
EXAMPLE = ROOT / "examples"
WEIGHTs = ROOT / "examples" / "weights"
# global logger
logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")


class TrackerVisualizer:
    def __init__(self):
        self.id_colors = {}

    def display_tracking_results(self, im, tracks, opt, thickness=2, fontscale=0.5):
        
        for track in tracks:
            xyxy = track[:4].astype(int)
            id, conf, cls = track[4].astype(int), track[5], track[6].astype(int)
            
            if id not in self.id_colors:
                self.id_colors[id] = random_bgr()
            color = self.id_colors[id]
            im = cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
            cv2.putText(im, f'id: {id}, conf: {conf:.2f}, class: {cls}', (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)
        cv2.imshow('Tracking Results', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True  
            
        return False
    