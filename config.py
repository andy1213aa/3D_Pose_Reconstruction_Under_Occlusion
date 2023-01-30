

class Config():
   
    cam_type = 'video'  # 'realsense', 'video', 'ipcam'
    cam_resolution_width = 1280 # (pixels)
    cam_resolution_height = 720  # pixels
    cam_frame_rate = 30  # fps
    cam_calibration_path = '/media/aaron/work/ITRI_SSTC/S200/anti_masking/realsense_new_4/calibration_realsense_new_4.json'
    
    merge_singleView_width = 640
    merge_singleView_height = 360
    merge_col_num = 2
    world_view = True

    # yolov5
    yolo_source_pth = '../yolov5'
    yolo_model_pth = '../yolov5n.engine'#'pt, engine, onnx'
    yolo_source_type = 'local'
    
    # mmpose 2D model path
    MM_pose_type = 'top_down' #'top_down, bottom_up'
    # TOP-DOWN
    # 'config_path' : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/hrnet_w48_coco_256x192.py'
    # 'ckpt_path'   : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    MM_config_path = '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/res50_coco_256x192.py'
    MM_ckpt_path = '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/res50_coco_256x192-ec54d7f3_20200709.pth'
    # MM_config_path = '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/res50_coco_256x192_fp16_dynamic.py'
    # MM_ckpt_path  = '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/res50_coco_256x192_fp16_dynamic-6edb79f3_20210430.pth'

    # BOTTOM-UP
    # 'config_path' : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/bottom_up/res50_coco_512x512.py'
    # 'ckpt_path'   : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/bottom_up/res50_coco_512x512-5521bead_20200816.pth'

    MM_smooth_cfg =  '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/smoother/one_euro.py'
    MM_cuda_idx = 0

    # Will affect the calculation speed
    merge_mmpose_width = 640
    merge_mmpose_height = 360
    merge_mmpose_col_num = 2

    #realsense serial numbers
    # Only work if "type" is "realsense"
    realsense_SN = [
        # '142422251208'
        '939622073590',
        '938422074078',
        '941322072583',
        '939622071867'
    ]

    # ipcam 
    # Only work if 'type' is 'ipcam'
    ipcam_rtsp = [
        'rtsp://admin:123456@192.168.0.3:554/profile1',
        'rtsp://admin:123456@192.168.0.73/stream0',
        'rtsp://admin:123456@192.168.0.123/stream0',
        'rtsp://admin:123456@192.168.0.233/stream0'
    ]
    # video_path 
    # Only wrok if "type" is 'video'
    video_folder = '/media/aaron/work/ITRI_SSTC/S200/anti_masking/realsense_new_4/720p/*.avi'
    
    #Camera idx pair
    cam_idx_pair = [[0, 1],
                    [1, 2]
                    ]
    # Denormalize the tvec to world coordinate.
    # Write in fixed value now.
    # Should be automatically calculate in future.               
    cam_denormalize = [264, 264, -170, -170]
        
    #Result evaluate
    record_result = () # ('2D'), ('3D') or ('2D', '3D')
    show_FPS = False
    show_kpt2D = False
    show_scene2D = True
    show_scene3D = False

