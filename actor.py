import numpy as np
import cv2 
from functools import wraps
import time
def measureExcutionTime(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time.time() - start
            print(f"{func.__name__} execution time: {end_: 0.4f} sec.")
    return _time_it

class actor():

    def __init__(self, pplIdxs, dataset):
        self.pplIdxs = pplIdxs # cross all cammera
        if dataset == 'coco':
            self.kpt_num = 17
            self.heatmap_height = 48
            self.heatmap_width = 64
        self.win_kpts3D = None
        self.win_cam_pair = None
   
    
    def set_winner_kpts(self, kpts3D, cams_idx):
        
        self.win_kpts3D = kpts3D.reshape(self.kpt_num, 3)
        self.win_cam_pair= cams_idx