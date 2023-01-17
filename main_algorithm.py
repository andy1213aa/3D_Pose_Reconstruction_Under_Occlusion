import numpy as np
import cv2
import utlis
import recover3Dpose
from actor import Actor

def get_match_list(frames:list,
                detection_model,
                pose_model,
                cam_nums:int,
                merge_mmpose_width:int,
                merge_mmpose_height:int,
                merge_mmpose_col_num=2,
                ):
        '''
        Stage 1: Matching. Get 2D pose information.
        '''
        # Stage1, get 2D pose information.
        detect_info = []

        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(
                frame,
                (merge_mmpose_width,merge_mmpose_height), 
                interpolation=cv2.INTER_NEAREST
            )
     
        '''
        "Yolo Detection"

        res: 1d nparray with only 1 or 0. Size equal to camera numbers. 
            -> 1 : people is detected in views.
            -> 0 : non of each one is detected. 

        yolo_result: 1d list of ndarray. The lens of the list is equal to the numbers of cameras.
        '''
        res, yolo_result = detection_model(frames)
        # Any peeson is detected in any view or not.
        if not res.any():
            return [], frames, detect_info
            
        kpt2D_frames, detect_info = utlis.batch_inference_top_down_mmpose(
            pose_model,
            frames,
            yolo_result,
            merge_mmpose_col_num,
            merge_mmpose_height,
            merge_mmpose_width,                                                        
            cam_nums
        )
        
        if not any(detect_info):
            return [], kpt2D_frames, detect_info
    
        '''
        Not implement ADMM yet.
        '''
        #先寫死，後續透過ADMM獲得
        single_idx = (1-res)
        single_idx[single_idx==1] = -1
        single_idx = [0]*cam_nums
        match_list = [single_idx]

        return match_list, kpt2D_frames, detect_info
        
def reconstruction_pose3D(cameras,
                        camera_pair,
                        pictoStruct,
                        match_list, 
                        detect_info):
        '''
        Stage2: reconstruction
        '''
        actors = []
        # Stage2, get 3D pose information 
        for pplIdxs in match_list:
         
            '''
            Step 1: calculate prior
            '''
            
            # Build actor class
            ppl = Actor(pplIdxs, 'coco')

            # Create candidate infomation
            candidates_cams, candidates_kpts, candidates_heatmap, candidates_intrinsic = recover3Dpose.get_candidate_info(ppl, 
                                                                                            cameras, 
                                                                                            detect_info, 
                                                                                            camera_pair) # (17, n, 3), (17, n, 64, 48)
            
            # If there exist at least two cameras get ppl infomation
            if len(candidates_cams) ==0:
                continue
      
            # Calculate heatmap prior
            heatmap_prior = recover3Dpose.get_heatmap_prior(candidates_cams, 
                                            candidates_kpts, 
                                            candidates_heatmap, 
                                            candidates_intrinsic, 
                                            pplIdxs, 
                                            detect_info)

            # Calculate bone lenght prior
            bone_length_prior = recover3Dpose.get_bone_length_prior(pictoStruct, 
                                                                candidates_kpts)
    
            '''
            Step 2: Calculate the maximum likelihood of prior  
            '''

            # heatmap_prior_prod = []
            # bone_length_prior_prod = []
            # total_prior_prod = []
            # for i in range(len(candidates_cams)):
            #     heatmap_prior_prod.append(np.prod(heatmap_prior[:, i])) 
            #     bone_length_prior_prod.append(np.prod(bone_length_prior[:, i]))
            #     total_prior_prod.append(np.prod(heatmap_prior[:, i]) * np.prod(bone_length_prior[:, i]))
                
  
            camera_pair_prior = [np.prod(heatmap_prior[:, i]) * np.prod(bone_length_prior[:, i]) for i in range(len(candidates_cams))]
            max_prior = np.argmax(camera_pair_prior)
            
            # print('Pair Camera: ', candidates_cams[max_prior])

            ppl.set_winner_kpts(candidates_kpts[:, max_prior, :], candidates_cams[max_prior])
            actors.append(ppl)

        return actors
