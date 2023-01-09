from kpts2D_model import Kpts2D_Model
from yolov5_model import yolov5_model
import pictorial_3D
import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera
import glob

def scene_initialize(self):

    self.init_cameras() 
    self.init_model()
    self.pictoStruct = pictorial_3D.init_3DPS()
    self.init_scene3D()
    self.cam_nums = len(self.cameras)

def init_cameras(self, camera_type):
    
    self.camera_type = self.cfg['type']
    self.camera_pair =  self.cfg['cam_idx_pair']

    match camera_type:

        case 'realsense':
            # check get camera or not
            connect_device = []
            for sn in self.cfg['realsense_SN']:
                for d in rs.context().devices:
                    if d.get_info(rs.camera_info.serial_number) == sn:
                        print('Found device: ',
                            d.get_info(rs.camera_info.name), ' ',
                            d.get_info(rs.camera_info.serial_number))
                        connect_device.append(d.get_info(rs.camera_info.serial_number))
            
            assert len(connect_device) >= 2, print('Registrition needs two camera connected.')

            for idx, device in enumerate(connect_device):
                self.cameras.append(Camera(self.cfg, device, idx))

        case 'video':
            videos = sorted(glob.glob(self.cfg['video_folder']))

            for idx, vname in enumerate(videos):
                video_reader = cv2.VideoCapture(vname)
                assert video_reader.isOpened(), f'Failed to read video...\nThe video path is: {vname}\n' 
                print(f'Successfully read video! \nThe video path is: {vname}\n')
                self.cameras.append(Camera(self.cfg, video_reader, idx))

        case 'ipcam':
            for idx, rtsp in enumerate(self.cfg['ipcam_rtsp']):
                ipcam_reader = cv2.VideoCapture(rtsp)
                assert ipcam_reader.isOpened(), 'IPcam rtsp address error...\nThe rtsp is:{rtsp}\n'
                print(f'Successfully read ipcam! \nThe rtsp path is: {rtsp}\n')
                self.cameras.append(Camera(self.cfg, ipcam_reader, idx))

def init_model(self):
        '''
        Load usage models and warm up.
        '''

        # Object detection model
        self.yolov5 = yolov5_model()
        print('Object detection model is loaded.')

        # pose2D detection model
        self.pose2D = Kpts2D_Model(self.cfg['config_path'],
                            self.cfg['ckpt_path'],
                            self.cfg['smooth_cfg'],
                            self.show_kpt2D,
                            self.cfg['pose_type'],
                            self.cfg['cuda_idx'],
                            )

        print('Pose 2D detection model is loaded.')

def init_scene3D(self):

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas_width, self.canvas_height = self.fig.get_size_inches() * self.fig.get_dpi()
        
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.view_init(elev = -157, azim=130)