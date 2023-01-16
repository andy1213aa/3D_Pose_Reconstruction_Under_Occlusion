import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import json
import threading
class Camera():
    
    def __init__(self, 
                resolution_width: int,
                resolution_height: int,
                frame_rate: int,
                camera_type:str,
                device: object, 
                idx: int):

        self.device = device
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.frame_rate = frame_rate
        self.camera_type = camera_type
        self.idx = idx
        self.calibration_parameter = None

        # ipcam
        self.record_single_view = True
        self.ipcam_update = True

        self.camera_initialize()
        self._now_frame_num = -1
        self._pre_frame_num = self._now_frame_num
        self.now_frame = None
        self.video_capture = None
        self.detect_info = {}
        
        if self.record_single_view:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.videoWriter = cv2.VideoWriter(f'cam_view_{self.idx}.mp4', 
                                            fourcc,frame_rate, 
                                            (resolution_width,  
                                            resolution_height))

    def camera_initialize(self, calibration_path):

        
        self.load_calibration_parameters(calibration_path)

        match self.camera_type:

            case 'realsense':
                # initialize the pipline
            
                rs_config = rs.config()
                rs_config.enable_stream(rs.stream.color, 
                                        self.resolution_width, 
                                        self.resolution_height, 
                                        rs.format.bgr8, 
                                        self.frame_rate)

                self.pipeline= rs.pipeline()
                rs_config.enable_device(self.device)
                self.pipeline.start(rs_config)

            case 'video':
                pass

            case 'ipcam':
                def update():
                    while self.ipcam_update:
                        ret, self.now_frame = self.device.read()
                        
                        current_time = time.asctime( time.localtime(time.time()) )
                        cv2.putText(self.now_frame, str(current_time), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        assert ret, 'Read ipcam stream fail...'
                        if self.record_single_view:
                            self.videoWriter.write(self.now_frame)
                self.video_capture_thread = threading.Thread(target=update, args=())
                self.video_capture_thread.daemon = True
                self.video_capture_thread.start()

    def read(self):
        
        '''
        Return a list of VideoCapture objects.
        '''
        match self.camera_type:

            case 'realsense':
                return self.pipeline.wait_for_frames()

            case 'video':
                success, frame = self.device.read()
                self.now_frame = frame
                assert success, 'Read video fail...'
                return frame

            case 'ipcam':
                return self.now_frame

    def stop(self):

        match self.camera_type:

            case 'realsense':
                self.pipeline.stop()
            case 'video':
                self.device.release()
            case 'ipcam':
                self.ipcam_update = False
                self.video_capture_thread.join()
                self.device.release()
        
        if self.record_single_view:
            self.videoWriter.release()

        
    def load_calibration_parameters(self, calibration_path):
       
        with open(calibration_path) as cfile:
            calib = json.load(cfile)

        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}
        
        # Convert data into numpy arrays for convenience
        
        for k,cam in cameras.items():
          
            if cam['name'] == f'00_{k[0]}{self.idx}':
                cam['K'] = np.matrix(cam['K'])
#                 cam['distCoef'] = np.array(cam['distCoef'])
                cam['R'] = np.matrix(cam['R'])
                cam['t'] = np.array(cam['t']).reshape((3,1))
                self.calibration_parameter = cam
            
        # movement
        self.movement = {}
        for mov in calib['movement']:
            tmp_movement = {}
            tmp_movement["R"] = np.array(mov["R"])
            tmp_movement["t"] = np.array(mov["t"])
            self.movement[mov["name"]] = tmp_movement

