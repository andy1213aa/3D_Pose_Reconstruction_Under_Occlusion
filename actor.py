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
   
    def get_candidate_info(self, videos, detect_info, camera_pair):
        
        '''
        output: 17xnx3
        '''
        
        cnt = 0
        candidates_intrinsic = np.zeros((self.kpt_num, len(camera_pair), 3, 3))
        candidates_kpts = np.zeros((self.kpt_num, len(camera_pair), 3, 1)) 
        candidates_heatmap = np.zeros((self.kpt_num, len(camera_pair), 64, 48))
        candidates_cams = []

        valid_candidates_idx = []  # some view pair have only one ppl is seen, then it is not valid

        for v1, v2 in camera_pair:
                
            if not detect_info[v1] or not detect_info[v2]:
                candidates_intrinsic[:, cnt] = np.empty((self.kpt_num, 3, 3))
                candidates_kpts[:, cnt] = np.empty((self.kpt_num, 3, 1))
                candidates_heatmap[:, cnt] = np.empty((self.kpt_num, 64, 48))
                continue
            
            # print('detect_info_len: ', len(detect_info))
            # print('detect_info_0: ' ,detect_info[0])
            # print('detect_info_0: ', detect_info[v2][self.pplIdxs[v2]])
            R_12 = videos[v1].movement[f'v{v1}v{v2}']['R']
            t_12 = videos[v1].movement[f'v{v1}v{v2}']['t']
    
            kpt2d_1 = detect_info[v1][self.pplIdxs[v1]]['keypoints'][:, :2]
            kpt2d_2 = detect_info[v2][self.pplIdxs[v2]]['keypoints'][:, :2]
            
            K_1 = videos[v1].calibration_parameter['K']
            K_2 = videos[v2].calibration_parameter['K']
            
            # print(f'  v{v1}v{v2}: ')
            # print(f'     R_12: {R_12}')
            # print(f'     r_12: {t_12}')
            # print(f'     kpt1: {kpt2d_1}')
            # print(f'     kpt2: {kpt2d_2}')
            
            points3D = self.triangulation(kpt2d_1, kpt2d_2, K_1, K_2, R_12, t_12)

            # print(f'     points3D: {points3D}')
            
            valid_candidates_idx.append(cnt)
            candidates_intrinsic[:, cnt] = K_1
            candidates_kpts[:, cnt] = points3D
            candidates_heatmap[:, cnt] =detect_info[v1][self.pplIdxs[v1]]['heatmap']
            candidates_cams.append([v1, v2])
            cnt += 1
          
        return candidates_cams, candidates_kpts[:, valid_candidates_idx, :], candidates_heatmap[:, valid_candidates_idx, :], candidates_intrinsic[:, valid_candidates_idx, :]
        # return self.candidates_kpts, self.candidates_heatmap


    def triangulation(self, kpt2d_a, kpt2d_b, K_a, K_b, R_ab, t_ab):
    
        M_a = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        M_b = np.hstack((R_ab, t_ab))

        camMat_a = np.dot(K_a,  M_a)
        camMat_b = np.dot(K_b,  M_b)

        point_4d_hom = cv2.triangulatePoints(camMat_a, camMat_b, np.expand_dims(kpt2d_a, axis=1), np.expand_dims(kpt2d_b, axis=1))

        points4D = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        points3D = np.expand_dims(points4D[:3, :].T, axis = -1)
     
        return points3D 
    def set_winner_kpts(self, kpts3D, cams_idx):
        
        self.win_kpts3D = kpts3D.reshape(self.kpt_num, 3)
        self.win_cam_pair= cams_idx