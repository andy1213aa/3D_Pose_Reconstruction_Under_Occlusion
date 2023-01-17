import numpy as np
import time
import pyrealsense2 as rs
from functools import wraps
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
    def __init__(self, 
                cameras: list,
                pplDetect_model,
                pose2D_model,
                pictoStruct: list,
                merge_singleView_width: int,
                merge_singleView_height: int,
                merge_col_num: int,
                cam_idx_pair: list,
                ):

        self.pplDetect_model = pplDetect_model
        self.pose2D_model = pose2D_model
        self.pictoStruct = pictoStruct
        self.merge_singleView_width = merge_singleView_width
        self.merge_singleView_height = merge_singleView_height
        self.merge_col_num = merge_col_num
        self.cameras = cameras
        self.cam_idx_pair = cam_idx_pair

        self.cam_nums = len(cameras)
        self.frame_num = 0

    def get_show_actors(self) -> list:

        actors = []
        self.frame_num += 1
        self.show_images = self.get_show_images(self.cameras[0].camera_type)
        print('---------------------------------------')
        print(f'Frame num: {self.frame_num}')
        
        # Stage1, get id matching and 2D pose (pixels)
        # match_list, detect_info = self.get_match_list(image_nparray)
        match_list, self.show_images, detect_info = main_algorithm.get_match_list(self.show_images,
                                                                                self.pplDetect_model,
                                                                                self.pose2D_model,
                                                                                self.cam_nums,
                                                                                self.merge_singleView_width,
                                                                                self.merge_singleView_height,
                                                                                self.merge_col_num)
        if match_list:
            actors = main_algorithm.reconstruction_pose3D(self.cameras,
                                                        self.cam_idx_pair,
                                                        self.pictoStruct,
                                                        match_list,
                                                        detect_info)

            # res = self.recover3Dpose(image_nparray, match_list, detect_info)

        return actors


    def get_show_images(self, camera_type) -> list:
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
