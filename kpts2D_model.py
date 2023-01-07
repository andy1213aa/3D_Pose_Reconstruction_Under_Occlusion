from mmpose.apis import (inference_top_down_pose_model,
                        init_pose_model,
                        vis_pose_result,
                        inference_bottom_up_pose_model)
import warnings
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
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
    
class Kpts2D_Model():

    def __init__(self, config_path, ckpt_path, smooth_cfg, visualization, pose_type, cuda_idx=0):
        
        
        '''
        pose 2D config and ckpt
        
        '''

        self.pose_type = pose_type
        pose_detector_config = config_path
        pose_detector_checkpoint = ckpt_path
        
        # gpu id
        device = f'cuda:{cuda_idx}'
        
        # initial pose model
        self.pose_det_model = init_pose_model(
            pose_detector_config,
            pose_detector_checkpoint,
            device=device)
        
        # initial smoother
        smooth_filter_cfg = smooth_cfg
        self.smoother2D = Smoother(filter_cfg=smooth_filter_cfg, keypoint_dim=2)
        
        # set dataset config
        self.pose_det_dataset = self.pose_det_model.cfg.data['test']['type']
        self.pose_dataset_info = self.pose_det_model.cfg.data['test'].get('dataset_info', None)
        if self.pose_dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
            assert (self.pose_det_dataset == 'BottomUpCocoDataset')
        else:
            self.pose_dataset_info = DatasetInfo(self.pose_dataset_info)
        
        # drawing parameter
        self.radius = 5
        self.thickness = 2
        self.visualization = visualization


    def __call__(self, frame, detect_result):
        '''
        Input:
            frame: raw frame data.
            detect_result: bbox detection result.
        '''
        detect_result = self.convert_object_detection_definition(detect_result)
    
        pose_det_results, heatmap = inference_top_down_pose_model(
                self.pose_det_model,
                frame,
                detect_result,
                bbox_thr=0.65,
                format='xyxy',
                dataset=self.pose_det_dataset,
                return_heatmap=True,
                outputs=None)


        # pose_det_results = self.smoother2D.smooth(pose_det_results)
        
        if self.visualization: 
            frame = vis_pose_result(
                self.pose_det_model,
                frame,
                pose_det_results,
                radius=self.radius,
                thickness=self.thickness,
                dataset=self.pose_det_dataset,
                dataset_info=self.pose_dataset_info,
                kpt_score_thr=0.3,
                show=False,
                out_file=None)
        
        return frame, pose_det_results, heatmap
 
    def convert_object_detection_definition(self, detect_result):
        
        # yolov5 converter

        #Deal with single frame
        
        if detect_result.shape[0] == 0:
            return []
        
        convert_result = []
        for bbox in detect_result:

            convert_result.append({'bbox': bbox[:-1]})
            
       
            
        return convert_result