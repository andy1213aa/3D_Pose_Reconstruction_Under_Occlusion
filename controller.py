import initialize
import pictorial_3D
import cv2




class StageManager():

    def _init__(self, config):
        self.cfg = config

        if self.record_result:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            MV = cv2.VideoWriter('multi_view.avi', fourcc, self.cfg['frame_rate'], (self.merge_singleView_width*2, self.merge_singleView_height*2))
            D3V = cv2.VideoWriter('D3view.avi', fourcc, self.cfg['frame_rate'], (int(self.canvas_width), int(self.canvas_height)))

    def initlize(self):
        self.cameras = initialize.init_cameras() 
        pplDetect_model, pose2D_model = initialize.init_model(self.cfg.config_path,
                                                            self.cfg.ckpt_path,
                                                            self.cfg.smooth_cfg,
                                                            self.cfg.pose_type,
                                                            self.cfg.cuda_idx)

        pictoStruct = initialize.init_3DPS()
        self.stage = initialize.init_scene(self.cameras,
                                    pplDetect_model,
                                    pose2D_model,
                                    pictoStruct,
                                    self.cfg.merge_singleView_width,
                                    self.cfg.merge_singleView_height,
                                    self.cfg.merge_col_num,
                                    self.cfg.cam_idx_pair
                                    )

        self.canvas, self.ax = initialize.init_view3D()
        
    
    def show_start(self):
        
        actors = self.stage.start()

        for cam in self.cameras:
            cam.stop()