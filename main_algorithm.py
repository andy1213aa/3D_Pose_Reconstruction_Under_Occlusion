import numpy as np
import cv2
import utlis
import recover3Dpose
import actor

def get_match_list(self, 
                frames:list):
        '''
        Stage 1: Matching. Get 2D pose information.
        '''
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
        
def reconstruction_pose3D(self, match_list, detect_info):
        '''
        Stage2: reconstruction
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
