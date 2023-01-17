import initialize
import cv2
from views import Visualization
from threading import Thread
import time

class StageManager():

    def __init__(self, config):
        self.cfg = config

        if '2D' in self.cfg.record_result:
            assert self.cfg.show_scene2D == True, 'The parameter "show_scene2D" in "config.py" should be set as "True" if you want to record the result.'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.MV = cv2.VideoWriter('multi_view.avi', fourcc, self.cfg['frame_rate'], (self.merge_singleView_width*2, self.merge_singleView_height*2))

        if '3D' in self.cfg.record_result:
            assert self.cfg.show_scene3D == True, 'The parameter "show_scene3D" in "config.py" should be set as "True" if you want to record the result.'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.D3V = cv2.VideoWriter('D3view.avi', fourcc, self.cfg['frame_rate'], (int(self.canvas_width), int(self.canvas_height)))

    def initlize(self):
        self.cameras = initialize.init_cameras(self.cfg.cam_type,
                                            self.cfg.cam_resolution_width,
                                            self.cfg.cam_resolution_height,
                                            self.cfg.cam_frame_rate,
                                            self.cfg.cam_calibration_path,
                                            realsense_SN = self.cfg.realsense_SN,
                                            video_folder = self.cfg.video_folder,
                                            ipcam_rtsp = self.cfg.ipcam_rtsp)

        self.pplDetect_model, self.pose2D_model = initialize.init_model(self.cfg.yolo_source_pth,
                                                            self.cfg.yolo_model_pth,
                                                            self.cfg.yolo_source_type,
                                                            self.cfg.MM_config_path,
                                                            self.cfg.MM_ckpt_path,
                                                            self.cfg.MM_smooth_cfg,
                                                            self.cfg.MM_pose_type,
                                                            self.cfg.MM_cuda_idx)

        pictoStruct = initialize.init_3DPS()
        self.stage = initialize.init_scene(self.cameras,
                                    self.pplDetect_model,
                                    self.pose2D_model,
                                    pictoStruct,
                                    self.cfg.merge_singleView_width,
                                    self.cfg.merge_singleView_height,
                                    self.cfg.merge_col_num,
                                    self.cfg.cam_idx_pair
                                    )


        self.canvas, self.ax = initialize.init_view3D()
  

        self.visualizer = Visualization(
                            cameras = self.cameras,
                            pose2D_model = self.pose2D_model,
                            show_FPS = self.cfg.show_FPS,
                            show_kpt2D = self.cfg.show_kpt2D,
                            world_view = self.cfg.world_view,
                            show_scene2D = self.cfg.show_scene2D,
                            show_scene3D = self.cfg.show_scene3D,
                            canvas = self.canvas,
                            ax = self.ax,
                            merge_col_num = self.cfg.merge_col_num,
                            merge_singleView_height = self.cfg.merge_singleView_height,
                            merge_singleView_width = self.cfg.merge_singleView_width,
                            cam_denormalize=self.cfg.cam_denormalize
                            )


    def show(self):
        print('show start!!')
        
        while True:
            start = time.time()
            actors = self.stage.get_show_actors()
            exec_time = time.time() - start
            print(f'FPS: {int(1 / (exec_time))}')
            self.visualizer.update_actor(actors)

            self.visualizer.update_show_images(self.stage.show_images)

            self.visualizer.update_exec_time(exec_time)
            
            # if '2D' in self.cfg.record_result:
            #     self.MV.write(self.visualizer.view2D)

            # if '3D' in self.cfg.record_result:
            #     self.D3V.write(self.visualizer.view3D)

    def start(self):
        
        vis_thread = Thread(target = self.visualizer.result_visualization,  args=())
        vis_thread.start()

        show_thread = Thread(target = self.show,  args=())
        show_thread.start()