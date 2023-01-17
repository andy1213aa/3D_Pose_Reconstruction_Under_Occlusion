import torch
import numpy as np

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
    
class yolov5_model():

    def __init__(self,
                source_pth,
                model_pth, 
                source_type):

        #yolo setting
        self.model = torch.hub.load(source_pth, 
                        'custom', 
                        path=model_pth, 
                        force_reload=True,
                        source=source_type,
                        )

        self.model.conf = 0.7         # NMS confidence threshold
        self.model.iou = 0.45          # NMS IoU threshold
        self.model.agnostic = False    # NMS class-agnostic
        self.model.multi_label = False # NMS multiple labels per box
        self.model.classes = [0]       # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        self.model.max_det = 1      # maximum number of detections per image
        self.model.amp = False         # Automatic Mixed Precision (AMP) inference

    # @measureExcutionTime
    def __call__(self, frames):
        res = np.zeros((len(frames)))
        # Maybe try to not move to cpu.
        
        # start = time.time()
        gpu_out = self.model(frames).xyxy
        # print(f'yolo gpu: {time.time()-start}')
        cpu_out = []

        for i, g in enumerate(gpu_out):
            tmp = g.cpu().numpy()
            if tmp.size > 0:
                res[i] = 1 
            cpu_out.append(tmp)
        # print(f'yolo to cpu: {time.time()-start}')
        return res, cpu_out


    

