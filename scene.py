import numpy as np
import pyrealsense2 as rs
import cv2
import time
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
from camera import Camera
import math
from actor import actor
from functools import wraps
from kpts2D_model import Kpts2D_Model
from yolov5_model import yolov5_model
from scipy import stats
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import utlis
import recover3Dpose
import pictorial_3D
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

    def scene_initialize(self):

        self.init_cameras() 
        self.init_model()
        self.pictoStruct = pictorial_3D.init_3DPS()
        self.init_scene3D()
        self.cam_nums = len(self.cameras)
        
    def init_cameras(self):
        self.camera_type = self.cfg['type']
        self.camera_pair =  self.cfg['cam_idx_pair']

        if self.camera_type == 'realsense':
            
            self.align = rs.align(rs.stream.color)
            # check get camera or not
            connect_device = []
            for sn in self.cfg['realsense_SN']:
                for d in rs.context().devices:
                    if d.get_info(rs.camera_info.serial_number) == sn:
                        print('Found device: ',
                            d.get_info(rs.camera_info.name), ' ',
                            d.get_info(rs.camera_info.serial_number))
                        connect_device.append(d.get_info(rs.camera_info.serial_number))
                    
            if len(connect_device) < 2:
                print('Registrition needs two camera connected.')
                exit()
  

            for idx, device in enumerate(connect_device):
                self.cameras.append(Camera(self.cfg, device, idx))

        elif self.camera_type == 'video':
            videos = sorted(glob.glob(self.cfg['video_folder']))

            for idx, vname in enumerate(videos):
                video_reader = cv2.VideoCapture(vname)
                assert video_reader.isOpened(), f'Failed to read video...\nThe video path is: {vname}\n' 
                print(f'Successfully read video! \nThe video path is: {vname}\n')
                self.cameras.append(Camera(self.cfg, video_reader, idx))

        elif self.camera_type == 'ipcam':
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
        


    def start(self):
        
        merge_col_num = self.cfg['merge_col_num']
        merge_singleView_height = self.cfg['merge_singleView_height']
        merge_singleView_width = self.cfg['merge_singleView_width']

        if self.record_result:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            MV = cv2.VideoWriter('multi_view.avi', fourcc, self.cfg['frame_rate'], (merge_singleView_width*2, merge_singleView_height*2))
            D3V = cv2.VideoWriter('D3view.avi', fourcc, self.cfg['frame_rate'], (int(self.canvas_width), int(self.canvas_height)))

        while True:
            self.actors = []
            start = time.time()
            self.frame_num += 1
            
            # No matter which type of camera, should have a list which collect frames with nparray type.
            image_nparray = []

            if self.camera_type == 'realsense':
                # wait for data
                frames = []
                for camera in self.cameras:
                    frames.append(camera.read())

                # align the input frame
                aligned_frames = []
                for f in frames:
                    aligned_frames.append(self.align.process(f))

                # grab those aligned frame
                aligned_color_frames = []
                for a_f in aligned_frames:
                    aligned_color_frames.append(a_f.get_color_frame())
    
                for a_c_f in aligned_color_frames:
                    image_nparray.append(np.asanyarray(a_c_f.get_data()))

                self.show_images = image_nparray.copy()

            elif self.camera_type == 'video':

                image_nparray = []
                for camera in self.cameras:
                    image_nparray.append(camera.read())

                self.show_images = image_nparray.copy()

            elif self.camera_type == 'ipcam':
                image_nparray = []
                for camera in self.cameras:
                    image_nparray.append(camera.read())

                self.show_images = image_nparray.copy()
            print('---------------------------------------')
            print(f'Frame num: {self.frame_num}')
            
            # Stage1, get id matching and 2D pose (pixels)
            match_list, detect_info = self.get_match_list(image_nparray)

            if match_list:
                res = self.recover3Dpose(image_nparray, match_list, detect_info)
            
            end = time.time()
            for i, view in enumerate(self.show_images):
          
                if self.show_FPS:
                    fps = int(1 / (end-start))
                    cv2.putText(view, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

                self.cameras[i].now_frame = view
            
            merge_multi_view = utlis.merge_images(self.show_images, merge_col_num, merge_singleView_height, merge_singleView_width)

            if self.record_result:
                MV.write(merge_multi_view)

            if self.show_scene3D:
                scene3D = self.scene_visualization()
                # scene3D = cv2.resize(scene3D, (merge_multi_view.shape[1], merge_multi_view.shape[0]), interpolation=cv2.INTER_NEAREST)
                # merge_multi_view = np.hstack((merge_multi_view, scene3D))
                cv2.imshow('scene3D', scene3D)
                if self.record_result:
                    D3V.write(scene3D)
            
            cv2.imshow('Views', merge_multi_view.astype(np.uint8))

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                for cam in self.cameras:
                    cam.stop()
                break

            elif key == ord('f') or key == ord('F'):
                self.show_FPS = not self.show_FPS

            elif key == ord('k') or key == ord('K'):
                self.show_kpt2D = not self.show_kpt2D
                self.pose2D.visualization = self.show_kpt2D

            elif key == ord('v') or key == ord('V'):
                self.show_scene3D = not self.show_scene3D
                if not self.show_scene3D:
                    cv2.destroyWindow('scene3D')
            # rotation
            elif key == ord('w') or key == ord('W'):
                self.ax.elev -= 1
            elif key == ord('a') or key == ord('A'):
                self.ax.azim -= 1
            elif key == ord('s') or key == ord('S'):
                self.ax.elev += 1
            elif key == ord('d') or key == ord('D'):
                self.ax.azim += 1

    @measureExcutionTime
    def get_match_list(self, frames:list):
       
        # Stage1, get 2D pose information.
        detect_info = []

        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(
                frame, 
                (self.cfg['merge_mmpose_width'],self.cfg['merge_mmpose_height']), 
                interpolation=cv2.INTER_NEAREST
            )
     
        '''
        "Yolo Detection"

        res: 1d nparray with only 1 or 0. Size equal to camera numbers. 
            -> 1 : people is detected in views.
            -> 0 : non of each one is detected. 

        yolo_result: 1d list of ndarray. The lens of the list is equal to the numbers of cameras.
        '''
        res, yolo_result = self.yolov5(frames)
        # Any peeson is detected in any view or not.
        if not res.any():
            return [], detect_info
            
        kpt2D_frames, detect_info = utlis.batch_inference_top_down_mmpose(
            self.pose2D,
            frames,
            yolo_result,
            self.cfg['merge_mmpose_col_num'],
            self.cfg['merge_mmpose_height'],
            self.cfg['merge_mmpose_width'],                                                        
            self.cam_nums
        )
        
        if not any(detect_info):
            return [], detect_info
        
        if self.show_kpt2D:
            self.show_images = kpt2D_frames

        '''
        Not implement ADMM yet.
        '''
        #先寫死，後續透過ADMM獲得
        white_ppl_idx = [2, 1, 1, 0, 0]
        red_ppl_idx = [1, 0, 2, 2, 1]
#         red_ppl_idx = [2,0,2,2,1]
        grey_ppl_idx = [0, -1, 0, 1, 2]

        single_idx = (1-res)
        single_idx[single_idx==1] = -1
        single_idx = [0]*self.cam_nums
   
        match_list = [single_idx]
        return match_list, detect_info

    @measureExcutionTime
    def recover3Dpose(self, frames:list, match_list, detect_info):
        '''
        Input:
            frames: list. Each element indicate a view with type "ndarray"
        '''
        
        # Stage2, get 3D pose information 
        for pplIdxs in match_list:
         
            '''
            Step 1: calculate prior
            '''
            
            # Build actor class
            ppl = actor(pplIdxs, 'coco')

            # Create candidate infomation
            # candidates_kpts, candidates_heatmap = ppl.get_candidate_info(self.cameras, detect_info) # (17, n, 3), (17, n, 64, 48)
            candidates_cams, candidates_kpts, candidates_heatmap, candidates_intrinsic = recover3Dpose.get_candidate_info(ppl, self.cameras, detect_info, self.camera_pair) # (17, n, 3), (17, n, 64, 48)
            
            # If there exist at least two cameras get ppl infomation
            if len(candidates_cams) ==0:
                continue
      
            # Calculate heatmap prior
            heatmap_prior = recover3Dpose.get_heatmap_prior(candidates_cams, candidates_kpts, candidates_heatmap, candidates_intrinsic, pplIdxs, detect_info)

            # Calculate bone lenght prior
            bone_length_prior = recover3Dpose.get_bone_length_prior(self.pictoStruct, candidates_kpts)
    
            '''
            Step 2: Calculate the maximum likelihood of prior  
            '''

            heatmap_prior_prod = []
            bone_length_prior_prod = []
            total_prior_prod = []
            for i in range(len(candidates_cams)):
                heatmap_prior_prod.append(np.prod(heatmap_prior[:, i])) 
                bone_length_prior_prod.append(np.prod(bone_length_prior[:, i]))
                total_prior_prod.append(np.prod(heatmap_prior[:, i]) * np.prod(bone_length_prior[:, i]))
                
  
            camera_pair_prior = [np.prod(heatmap_prior[:, i]) * np.prod(bone_length_prior[:, i]) for i in range(len(candidates_cams))]
            max_prior = np.argmax(camera_pair_prior)
            
            print('Pair Camera: ', candidates_cams[max_prior])

            ppl.set_winner_kpts(candidates_kpts[:, max_prior, :], candidates_cams[max_prior])
            self.actors.append(ppl)

        return 1

    @measureExcutionTime
    def scene_visualization(self):

        # self.ax.axis('off')

        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]
# 3D
#         skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
#                 [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
#                 [8, 14], [14, 15], [15, 16]]
        # 棋盤格
        width = 7
        height = 5
        objp = np.zeros((height*width,3), np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
        objp *= 6.8

        hd_cameras = [self.cam_info(self.cameras[i].calibration_parameter['R'], 
                       self.cameras[i].calibration_parameter['t']) for i in range(len(self.cameras))]

        if self.cfg['world_view']:
            # Draw selected camera subset in blue   
            for cam in hd_cameras:
                cc = -cam['R'].transpose()@cam['t']
                self.ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,0,1], s=10)
                
            # Draw chessboard
            for objpt in objp:
                self.ax.scatter(objpt[0], objpt[1], objpt[2], '.', color=[1,0,0], s= 1)

        if self.actors:
    
            for ppl in self.actors:
    #             kpt3d_world = self.convert_keypoint_definition(ppl.win_kpts3D)
                
                # Denormalize
                kpt3d_world = ppl.win_kpts3D * self.cfg['cam_denormalize'][ppl.win_cam_pair[0]]
                # transformation to world coordination
                if self.cfg['world_view']:
                    kpt3d_world = self.convert_keypoint_cam2world(kpt3d_world, ppl.win_cam_pair)

                for objpt in kpt3d_world:
                    self.ax.scatter(objpt[0], objpt[1], objpt[2], '.', color=[1,0,0], s=0.5)

                for skl in skeleton:
                    self.ax.plot([kpt3d_world[skl[0]][0], kpt3d_world[skl[1]][0]], 
                        [kpt3d_world[skl[0]][1], kpt3d_world[skl[1]][1]], 
                        [kpt3d_world[skl[0]][2], kpt3d_world[skl[1]][2]])

        self.canvas.draw()       # draw the canvas, cache the renderer
        scene_pltshowing = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(self.canvas_height), 
                                                                                            int(self.canvas_width), 3) 

        # cv2.imwrite(f'vis/test_{self.frame_num}.png', scene_pltshowing)
        self.ax.cla()
        return scene_pltshowing

    
    def cam_info(self, rvec, tvec):
        cam = {
            'R': rvec,
            't': tvec
        }
        return cam

    def convert_keypoint_definition(self, keypoints):
        """Convert pose det dataset keypoints definition to pose lifter dataset
        keypoints definition.

        Args:
            keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
            pose_det_dataset, (str): Name of the dataset for 2D pose detector.
            pose_lift_dataset (str): Name of the dataset for pose lifter model.
        """
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new

    def convert_keypoint_cam2world(self, keypoints, cams_idx):

        keypoints = keypoints.reshape(17, 3, 1)
        R_W = self.cameras[cams_idx[0]].calibration_parameter['R']
        t_W = self.cameras[cams_idx[0]].calibration_parameter['t']
        world_kpt3d = np.zeros((keypoints.shape[0], 3, 1))
        for i, kpt3D in enumerate(keypoints):
            world_kpt3d[i] =R_W.T@(kpt3D - t_W)

        world_kpt3d[:, 2] -= world_kpt3d[-1, 2]
        # 先寫死
        world_kpt3d = world_kpt3d.reshape((world_kpt3d.shape[0], 3))

        if cams_idx[0] == 2:
            RR = np.array([[    0.98472,   -0.073716,     0.15775],
                            [   0.066247,     0.99644,    0.052097],
                            [   -0.16103,   -0.040851,      0.9861]])
                            
            tt = np. array([[    -36.759],
                            [    -6.3854],
                            [    -7.7052]])

            world_kpt3d = (RR@world_kpt3d.T).T + tt.reshape(3, )

        return world_kpt3d