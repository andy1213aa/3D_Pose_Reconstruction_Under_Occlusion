


video_infomation = {

    'type' : 'video',  # 'realsense', 'video', 'ipcam'
    'resolution_width' : 1280,  # (pixels)
    'resolution_height' : 720,  # pixels
    'frame_rate' : 30,  # fps
    'merge_singleView_width': 640,
    'merge_singleView_height': 360,
    'merge_col_num': 2,
    'world_view': True,
    'calibration_path': '/media/aaron/work/ITRI_SSTC/S200/anti_masking/realsense_new_4/calibration_realsense_new_4.json',


    # mmpose 2D model path
    # 'config_path' : f'F:/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/hrnet_w48_coco_256x192.py',
    # 'ckpt_path'   : f'F:/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',

    'config_path' : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/res50_coco_256x192.py',
    'ckpt_path'   : '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/res50_coco_256x192-ec54d7f3_20200709.pth',

    # 'config_path' : f'F:/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/configs/Top_down/mobilenetv2_coco_256x192.py',
    # 'ckpt_path'   : f'F:/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/2D/ckpts/Top_down/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth',
    'smooth_cfg' :  '/media/aaron/work/ITRI_SSTC/S200/golf/MM/mmpose_test/3D_pose_sview_multiperson/mmpose/smoother/one_euro.py',
    'cuda_idx': 1,

    # yolov5
    'yolov5_path': '../yolov5s.pt',

    #realsense serial numbers
    # Only work if "type" is "realsense"
    'realsense_SN' : [
        '939622073590',
        '938422074078',
        '941322072583',
        '939622071867'
    ],

    # ipcam 
    # Only work if 'type' is 'ipcam'
    'ipcam_rtsp' :[
        'rtsp://admin:123456@192.168.0.3:554/profile1',
        'rtsp://admin:123456@192.168.0.73/stream0',
        'rtsp://admin:123456@192.168.0.123/stream0',
        'rtsp://admin:123456@192.168.0.233/stream0'
    ],
    # video_path 
    # Only wrok if "type" is 'video'
    'video_folder': '/media/aaron/work/ITRI_SSTC/S200/anti_masking/realsense_new_4/720p/*.avi',
    
    #Camera idx pair
    'cam_idx_pair': [[0, 1],
                     [1, 2]],
    # Denormalize the tvec to world coordinate.
    # Write in fixed value now.
    # Should be automatically calculate in future.               
    'cam_denormalize': [264, 264, -170, -170],

}


