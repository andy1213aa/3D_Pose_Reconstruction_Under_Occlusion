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
    
    def __init__(self, config: dict, device: object, idx: int):
        self.device = device
        self.cfg = config
        self.calibration_path = config['calibration_path']
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
                                            fourcc,self.cfg['frame_rate'], 
                                            (self.cfg['resolution_width'],  
                                            self.cfg['resolution_height']))

    def camera_initialize(self):

        
        self.load_calibration_parameters()

        if self.cfg['type'] == 'realsense':
            # initialize the pipline
            
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, 
                                    self.cfg['resolution_width'], 
                                    self.cfg['resolution_height'], 
                                    rs.format.bgr8, 
                                    self.cfg['frame_rate'])

            self.pipeline= rs.pipeline()
            rs_config.enable_device(self.device)
            self.pipeline.start(rs_config)

        elif self.cfg['type'] == 'video':
            pass
        elif self.cfg['type'] == 'ipcam':
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
        if self.cfg['type'] == 'video':
            success, frame = self.device.read()
            self.now_frame = frame
            assert success, 'Read video fail...'
            return frame

        elif self.cfg['type'] == 'realsense':
            return self.pipeline.wait_for_frames()

        elif self.cfg['type'] == 'ipcam':
            
            return self.now_frame

    def stop(self):

        if self.cfg['type'] == 'realsense':
            self.pipeline.stop()

        elif self.cfg['type'] == 'video':
            self.device.release()
            
        elif self.cfg['type'] == 'ipcam':
            self.ipcam_update = False
            self.video_capture_thread.join()
            self.device.release()
        
        if self.record_single_view:
            self.videoWriter.release()

        
    def load_calibration_parameters(self):
       
        with open(self.calibration_path) as cfile:
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

