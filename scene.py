import numpy as np

import cv2
import time
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from functools import wraps
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import utlis
import main_algorithm
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


class scene():
    def __init__(self, config):
        self.cfg = config
        self.cameras = []        
        self.frame_num = 0

        # record result
        self.record_result = False
        # show parameter
        self.show_FPS = False
        self.show_kpt2D = False
        self.show_scene3D = False

    def start(self):
        
        merge_col_num = self.cfg['merge_col_num']
        merge_singleView_height = self.cfg['merge_singleView_height']
        merge_singleView_width = self.cfg['merge_singleView_width']

        if self.record_result:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            MV = cv2.VideoWriter('multi_view.avi', fourcc, self.cfg['frame_rate'], (merge_singleView_width*2, merge_singleView_height*2))
            D3V = cv2.VideoWriter('D3view.avi', fourcc, self.cfg['frame_rate'], (int(self.canvas_width), int(self.canvas_height)))

        self.actors = []
        self.frame_num += 1
        
        self.show_images = self.get_show_images(self.camera_type)
        print('---------------------------------------')
        print(f'Frame num: {self.frame_num}')
        
        # Stage1, get id matching and 2D pose (pixels)
        # match_list, detect_info = self.get_match_list(image_nparray)
        match_list, self.show_images, detect_info = main_algorithm.get_match_list(self.show_images,
                                                                                self.yolov5,
                                                                                self.pose2D,
                                                                                self.cam_nums,
                                                                                merge_singleView_width,
                                                                                merge_singleView_height,
                                                                                merge_col_num)
        if match_list:
            self.actors = main_algorithm.reconstruction_pose3D(self.cameras,
                                                            self.camera_pair,
                                                            self.pictoStruct,
                                                            match_list,
                                                            detect_info)

            # res = self.recover3Dpose(image_nparray, match_list, detect_info)


    def cam_info(self, rvec, tvec):
        cam = {
            'R': rvec,
            't': tvec
        }
        return cam

    def get_show_images(self, camera_type):
        # No matter which type of camera, should have a list which collect frames with nparray type.
        image_nparray = []

        match camera_type:
            case 'realsense':
                # wait for data
                frames = []
                align = rs.align(rs.stream.color)
                for camera in self.cameras:
                    frames.append(camera.read())

                # align the input frame
                aligned_frames = []
                for f in frames:
                    aligned_frames.append(align.process(f))

                # grab those aligned frame
                aligned_color_frames = []
                for a_f in aligned_frames:
                    aligned_color_frames.append(a_f.get_color_frame())
    
                for a_c_f in aligned_color_frames:
                    image_nparray.append(np.asanyarray(a_c_f.get_data()))

            case 'video':
                for camera in self.cameras:
                    image_nparray.append(camera.read())


            
            case 'ipcam':
                for camera in self.cameras:
                    image_nparray.append(camera.read())

           

        return image_nparray
            