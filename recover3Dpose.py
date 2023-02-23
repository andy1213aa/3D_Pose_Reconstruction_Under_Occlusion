import numpy as np
from scipy import stats
import time
from functools import wraps
import cv2

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

def get_heatmap_prior(candidates_cams, 
                    candidates_kpts, 
                    candidates_heatmap, 
                    candidates_intrinsic, 
                    pplIdxs, 
                    detect_info):
            
        # 計算每個關節點
        total_prior = []

        candidates_reProject2d_homo = candidates_intrinsic @ candidates_kpts #(17, cam_num_composition, 3, 3) x (17, cam_num_composition, 3, 1) = (17, cam_num_composition, 3, 1)

        for joint_idx in range(candidates_kpts.shape[0]): 
            same_kpts_different_view = []
            for camera_pair in range(candidates_kpts.shape[1]): # 依照相機數量
                
    
                camID = candidates_cams[camera_pair][0]
                actorID = pplIdxs[camID]

                cropShape = (int(detect_info[camID][actorID]['bbox'][3] - detect_info[camID][actorID]['bbox'][1]),
                             int(detect_info[camID][actorID]['bbox'][2] - detect_info[camID][actorID]['bbox'][0]),
                            3)

                reProject2d_homo = candidates_reProject2d_homo[joint_idx][camera_pair]
                reProject2d_homo /= reProject2d_homo[2]
                reProject2d = reProject2d_homo[:2]

                '''
                heatmap
                '''

                shift = [detect_info[camID][actorID]['bbox'][0], 
                         detect_info[camID][actorID]['bbox'][1]]
                joint_heatmap = cv2.resize(candidates_heatmap[joint_idx][camera_pair], 
                                           (cropShape[1], cropShape[0]), interpolation=cv2.INTER_NEAREST)

                maxi = np.max(joint_heatmap)
                mini = np.min(joint_heatmap)
          
            
                joint_heatmap = (joint_heatmap -mini) / (maxi - mini)
                
                #convert coordinaion
                reProject2d[0] -= shift[0]
                reProject2d[1] -= shift[1]
                
                if reProject2d[0] < 0 or reProject2d[1] < 0 or reProject2d[0] > cropShape[1] or reProject2d[1] > cropShape[0]:
                    same_kpts_different_view.append(1e-6)
                else:
        #       #calculate heatmap value
                    probability = joint_heatmap[int(reProject2d[1]), int(reProject2d[0])]
                    same_kpts_different_view.append(probability)

                

            total_prior.append(same_kpts_different_view)
        return np.array(total_prior)


def get_bone_length_prior(edges, 
                        candidates_kpts):
        
        coco2h36m = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        '''
        Points3D should be convert to h36m index first.
        '''
        convert_definition_candidates_kpts = candidates_kpts[coco2h36m]

        bone_length_prior_result = []
        for edge in edges:
            bone_length = np.linalg.norm(convert_definition_candidates_kpts[edge['child']] 
                                         - convert_definition_candidates_kpts[edge['parent']], axis = 1)
            relative_error = np.abs ( bone_length - edge['bone_mean'] ) / edge['bone_std']
            
            prior = stats.norm.sf ( relative_error ) * 2
   
            bone_length_prior_result.append(prior)

        return np.array(bone_length_prior_result)
        
def get_candidate_info(actor, 
                    cameras, 
                    detect_info, 
                    camera_pair):
        
        '''
        output: 17xnx3
        ''' 
        
        cnt = 0
        candidates_intrinsic = np.zeros((actor.kpt_num, len(camera_pair), 3, 3))
        candidates_kpts = np.zeros((actor.kpt_num, len(camera_pair), 3, 1)) 
        candidates_heatmap = np.zeros((actor.kpt_num, len(camera_pair), 64, 48))
        candidates_cams = []

        valid_candidates_idx = []  # some view pair have only one ppl is seen, then it is not valid

        for v1, v2 in camera_pair:
                
            if not detect_info[v1] or not detect_info[v2]:
                candidates_intrinsic[:, cnt] = np.empty((actor.kpt_num, 3, 3))
                candidates_kpts[:, cnt] = np.empty((actor.kpt_num, 3, 1))
                candidates_heatmap[:, cnt] = np.empty((actor.kpt_num, 64, 48))
                continue

            R_12 = cameras[v1].movement[f'v{v1}v{v2}']['R']
            t_12 = cameras[v1].movement[f'v{v1}v{v2}']['t']
    
            kpt2d_1 = detect_info[v1][actor.pplIdxs[v1]]['keypoints'][:, :2]
            kpt2d_2 = detect_info[v2][actor.pplIdxs[v2]]['keypoints'][:, :2]
            
            K_1 = cameras[v1].calibration_parameter['K']
            K_2 = cameras[v2].calibration_parameter['K']
            
            
            points3D = triangulation(kpt2d_1, kpt2d_2, K_1, K_2, R_12, t_12)
            
            valid_candidates_idx.append(cnt)
            candidates_intrinsic[:, cnt] = K_1
            candidates_kpts[:, cnt] = points3D
            candidates_heatmap[:, cnt] =detect_info[v1][actor.pplIdxs[v1]]['heatmap']
            candidates_cams.append([v1, v2])
            cnt += 1
          
        return candidates_cams, candidates_kpts[:, valid_candidates_idx, :], candidates_heatmap[:, valid_candidates_idx, :], candidates_intrinsic[:, valid_candidates_idx, :]

def triangulation( 
                kpt2d_a, 
                kpt2d_b, 
                K_a, 
                K_b, 
                R_ab, 
                t_ab):

    M_a = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    M_b = np.hstack((R_ab, t_ab))

    camMat_a = np.dot(K_a,  M_a)
    camMat_b = np.dot(K_b,  M_b)

    point_4d_hom = cv2.triangulatePoints(camMat_a, camMat_b, np.expand_dims(kpt2d_a, axis=1), np.expand_dims(kpt2d_b, axis=1))

    points4D = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    points3D = np.expand_dims(points4D[:3, :].T, axis = -1)
    
    return points3D 