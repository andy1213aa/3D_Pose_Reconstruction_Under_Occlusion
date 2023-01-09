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

@measureExcutionTime
def batch_inference_top_down_mmpose(model, 
                                    frames, 
                                    yolo_result, 
                                    col_num, 
                                    merge_singleView_height, 
                                    merge_singleView_width, 
                                    seperate_num):
    '''
    MMpose not support batch inference.
    So we merge multiple view images into one to do like a batch inference way.

    merge_kpt2D_vis: list of dict. Each dict have two keys: bbox and keypoints.
        -> bbox: 1D ndarray
        -> keypoints: 2D ndarray. (Probably 17x3)
    '''

    merge_frames = merge_images(frames, col_num, merge_singleView_height, merge_singleView_width)
    merge_yolo_result= shift_yolo_bbox(yolo_result, col_num, merge_singleView_height, merge_singleView_width)
    merge_kpt2D_vis, merge_kpt2D, merge_heatmap = model(merge_frames, merge_yolo_result)
    kpt2D_frames = undo_merge_image(merge_kpt2D_vis,col_num, merge_singleView_height, merge_singleView_width) 
    detect_info = unshift_merge_top_down_kpt2D(merge_kpt2D, merge_heatmap, col_num, merge_singleView_height, merge_singleView_width, seperate_num)

    return kpt2D_frames, detect_info

def batch_inference_bottom_up_mmpose(model, 
                                    frames,  
                                    col_num, 
                                    merge_singleView_height, 
                                    merge_singleView_width, 
                                    seperate_num):
    merge_frames = merge_images(frames, col_num, merge_singleView_height, merge_singleView_width)
    merge_kpt2D_vis, merge_kpt2D, merge_heatmap = model(merge_frames, [])
    # if not merge_kpt2D.any():
    #     return merge_kpt2D_vis, []
    kpt2D_frames = undo_merge_image(merge_kpt2D_vis,col_num, merge_singleView_height, merge_singleView_width)
    detect_info = unshift_merge_bottom_up_kpt2D(merge_kpt2D, merge_heatmap, col_num, merge_singleView_height, merge_singleView_width, seperate_num)
    return kpt2D_frames, detect_info

@measureExcutionTime
def merge_images(images: list, col_num: int, merge_singleView_height: int, merge_singleView_width:int):

    '''
    Merge multiple images together.
    '''
    
    merge_multi_view = np.zeros((int(np.ceil(len(images)/col_num) * merge_singleView_height), 
                                        col_num * merge_singleView_width, 3), 
                                        dtype='uint8')
    
    for i, view in enumerate(images):
          
        row_idx = i // col_num 
        col_idx = i % col_num

        _view = cv2.resize(view, 
                            (merge_singleView_width, merge_singleView_height), 
                            interpolation=cv2.INTER_NEAREST)

        merge_multi_view[row_idx*merge_singleView_height:(row_idx+1)*merge_singleView_height, 
                        col_idx*merge_singleView_width:(col_idx+1)*merge_singleView_width] = _view

    return merge_multi_view

@measureExcutionTime
def shift_yolo_bbox(yolo_batch_result:list, col_num, merge_singleView_height, merge_singleView_width) -> np.ndarray:

    '''
    yolo_batch_result: list of ndarray.

    Return: ndarray
    '''
    result = []


    for view, bboxes in enumerate(yolo_batch_result):
        
        row_idx = view // col_num 
        col_idx = view % col_num
        
        for bbox in bboxes:

            # X
            bbox[0] += col_idx * merge_singleView_width
            bbox[2] += col_idx * merge_singleView_width

            # Y
            bbox[1] += row_idx * merge_singleView_height
            bbox[3] += row_idx * merge_singleView_height
            result.append(bbox)
            
    return np.array(result)

@measureExcutionTime
def undo_merge_image(merge_image: np.ndarray, col_num: int, merge_singleView_height: int, merge_singleView_width:int):
    merge_height, merge_width, _= merge_image.shape
    assert merge_height % merge_singleView_height == 0 and merge_width % merge_singleView_width == 0, 'Merge size is fail, please check again.'
    seperate_view_nums = int((merge_height / merge_singleView_height) * (merge_width / merge_singleView_width))
    result = []
    for i in range(seperate_view_nums):
        
        row_idx = i // col_num 
        col_idx = i % col_num

        tmp_seperate_image = merge_image[row_idx*merge_singleView_height:(row_idx+1)*merge_singleView_height, 
                        col_idx*merge_singleView_width:(col_idx+1)*merge_singleView_width]
        result.append(tmp_seperate_image)

    return result

@measureExcutionTime
def unshift_merge_top_down_kpt2D(merge_kpt2D: list, merge_heatmap, col_num: int, merge_singleView_height: int, merge_singleView_width:int, seperate_num:int)->list:

    result = [[] for _ in range(seperate_num)]
    
    for i, detect_info in enumerate(merge_kpt2D):
        
        '''
        Use top-left anchor to check which original idx the bbox is.
        '''

        row_idx = detect_info['bbox'][1] // merge_singleView_height
        col_idx = detect_info['bbox'][0] // merge_singleView_width
        
        original_idx = int(row_idx * col_num + col_idx)

        '''
        unshift coordination
        '''
        # bbox X
        detect_info['bbox'][0] -= col_idx * merge_singleView_width
        detect_info['bbox'][2] -= col_idx * merge_singleView_width

        # bbox Y
        detect_info['bbox'][1] -= row_idx * merge_singleView_height
        detect_info['bbox'][3] -= row_idx * merge_singleView_height

        # keypoints X
        detect_info['keypoints'][:, 0] -=  col_idx * merge_singleView_width
        # keypoints Y
        detect_info['keypoints'][:, 1] -=  row_idx * merge_singleView_height

        #heatmap
        detect_info['heatmap'] = merge_heatmap[0]['heatmap'][i]

        result[original_idx].append(detect_info)
        

    # assert len(result) == len(ppl_num_ineach_view), "Error accure with kpt unshift."
    return result
    
@measureExcutionTime
def unshift_merge_bottom_up_kpt2D(merge_kpt2D: list, merge_heatmap, col_num: int, merge_singleView_height: int, merge_singleView_width:int, seperate_num:int)->list:

    result = [[] for _ in range(seperate_num)]
    
    for i, detect_info in enumerate(merge_kpt2D):
        
        '''
        Use top-left anchor to check which original idx the bbox is.
        '''

        row_idx = detect_info['keypoints'][0, 1] // merge_singleView_height
        col_idx = detect_info['keypoints'][0, 0] // merge_singleView_width
        
        original_idx = int(row_idx * col_num + col_idx)

        '''
        unshift coordination
        '''

        # keypoints X
        detect_info['keypoints'][:, 0] -=  col_idx * merge_singleView_width
        # keypoints Y
        detect_info['keypoints'][:, 1] -=  row_idx * merge_singleView_height

        #heatmap
        detect_info['heatmap'] = merge_heatmap[0]['heatmap'][i]

        result[original_idx].append(detect_info)
        

    # assert len(result) == len(ppl_num_ineach_view), "Error accure with kpt unshift."
    return result