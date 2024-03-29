from kpts2D_model import Kpts2D_Model
from yolov5_model import yolov5_model
import pyrealsense2 as rs
import cv2
from camera import Camera
import glob
import matplotlib.pyplot as plt
import scene
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

def init_cameras(camera_type:str,
                resolution_width:int,
                resolution_height:int,
                frame_rate:int,
                calibration_path:str,
                realsense_SN = None,
                video_folder = None,
                ipcam_rtsp = None) -> list:
    '''
    Return List of Camera Object.
    '''
    
    cameras = []
    match camera_type:

        case 'realsense':
            # check get camera or not
            connect_device = []
            for sn in realsense_SN:
                for d in rs.context().devices:
                    if d.get_info(rs.camera_info.serial_number) == sn:
                        print('Found device: ',
                            d.get_info(rs.camera_info.name), ' ',
                            d.get_info(rs.camera_info.serial_number))
                        connect_device.append(d.get_info(rs.camera_info.serial_number))
            
            assert len(connect_device) >= 2, print('Registrition needs two camera connected.')

            for idx, device in enumerate(connect_device):
                cameras.append(Camera(resolution_width,
                                    resolution_height,
                                    frame_rate,
                                    calibration_path,
                                    camera_type,
                                    device, 
                                    idx))

        case 'video':
            
            videos = sorted(glob.glob(video_folder))

            for idx, vname in enumerate(videos):
                video_reader = cv2.VideoCapture(vname)
                assert video_reader.isOpened(), f'Failed to read video...\nThe video path is: {vname}\n' 
                print(f'Successfully read video! \nThe video path is: {vname}\n')
                cameras.append(Camera(resolution_width,
                                    resolution_height, 
                                    frame_rate,
                                    calibration_path,
                                    camera_type,
                                    video_reader, 
                                    idx))

        case 'ipcam':
            for idx, rtsp in enumerate(ipcam_rtsp):
                ipcam_reader = cv2.VideoCapture(rtsp)
                assert ipcam_reader.isOpened(), 'IPcam rtsp address error...\nThe rtsp is:{rtsp}\n'
                print(f'Successfully read ipcam! \nThe rtsp path is: {rtsp}\n')
                cameras.append(Camera(resolution_width,
                                    resolution_height, 
                                    frame_rate,
                                    calibration_path,
                                    camera_type,
                                    ipcam_reader, 
                                    idx))                
         

    return cameras
def init_model(yolo_source_pth,
            yolo_model_pth,
            yolo_source_type,
            MM_config_path,
            MM_ckpt_path,
            MM_smooth_cfg,
            MM_pose_type,
            MM_cuda_idx):
    '''
    Load usage models.
    '''

    # Object detection model
    # print(f'yolo_source_pth: {yolo_source_pth}')
    yolov5 = yolov5_model(yolo_source_pth,
                yolo_model_pth, 
                yolo_source_type)
    print('Object detection model is loaded.')

    # pose2D detection model
    pose2D = Kpts2D_Model(MM_config_path,
                        MM_ckpt_path,
                        MM_smooth_cfg,
                        MM_pose_type,
                        MM_cuda_idx,)

    print('Pose 2D detection model is loaded.')
    return yolov5, pose2D
    
def init_view3D():

    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev = -157, azim=130)

    return canvas, ax

def init_scene(cameras: list,
                pplDetect_model,
                pose2D_model,
                pictoStruct: list,
                merge_singleView_width: int,
                merge_singleView_height: int,
                merge_col_num: int,
                cam_idx_pair: list):

    return scene.scene(cameras,
                pplDetect_model,
                pose2D_model,
                pictoStruct,
                merge_singleView_width,
                merge_singleView_height,
                merge_col_num,
                cam_idx_pair)




def init_3DPS() -> list:
        def load_distribution(dataset='Unified'):
            joints2edges = {(0, 1): 0,
                            (1, 0): 0,
                            (0, 2): 1,
                            (2, 0): 1,
                            (0, 7): 2,
                            (7, 0): 2,
                            (0, 8): 3,
                            (8, 0): 3,
                            (1, 3): 4,
                            (3, 1): 4,
                            (2, 4): 5,
                            (4, 2): 5,
                            (3, 5): 6,
                            (5, 3): 6,
                            (4, 6): 7,
                            (6, 4): 7,
                            (7, 9): 8,
                            (9, 7): 8,
                            (8, 10): 9,
                            (10, 8): 9,
                            (9, 11): 10,
                            (11, 9): 10,
                            (10, 12): 11,
                            (12, 10): 11}
            distribution_dict = {
                'Shelf': {'mean': np.array ( [0.30280354, 0.30138756, 0.79123502, 0.79222949, 0.28964179,
                                            0.30393598, 0.24479075, 0.24903801, 0.40435882, 0.39445121,
                                            0.3843522, 0.38199836] ),
                        'std': np.array ( [0.0376412, 0.0304385, 0.0368604, 0.0350577, 0.03475468,
                                            0.03876828, 0.0353617, 0.04009757, 0.03974647, 0.03696424,
                                            0.03008979, 0.03143456] ) * 2, 'joints2edges': joints2edges
                        },
                'Campus':
                    {'mean': np.array ( [0.29567343, 0.28090078, 0.89299809, 0.88799211, 0.32651703,
                                        0.33454941, 0.29043165, 0.29932416, 0.43846395, 0.44881553,
                                        0.46952846, 0.45528477] ),
                    'std': np.array ( [0.01731019, 0.0226062, 0.06650426, 0.06009805, 0.04606478,
                                        0.04059899, 0.05868499, 0.06553948, 0.04129285, 0.04205624,
                                        0.03633746, 0.02889456] ) * 2, 'joints2edges': joints2edges},
                'Unified': {'mean': np.array ( [0.29743698, 0.28764493, 0.86562234, 0.86257052, 0.31774172,
                                                0.32603399, 0.27688682, 0.28548218, 0.42981244, 0.43392589,
                                                0.44601327, 0.43572195] ),
                            'std': np.array ( [0.02486281, 0.02611557, 0.07588978, 0.07094158, 0.04725651,
                                            0.04132808, 0.05556177, 0.06311393, 0.04445206, 0.04843436,
                                            0.0510811, 0.04460523] ) * 16, 'joints2edges': joints2edges}
            }
            # logger.debug ( f"Using distribution on {dataset}" )
            return distribution_dict[dataset]

        def getskel():
            skel = {}
            skel['tree'] = [{} for i in range ( 13 )]
            skel['tree'][0]['name'] = 'Nose'
            skel['tree'][0]['children'] = [1, 2, 7, 8]
            skel['tree'][1]['name'] = 'LSho'
            skel['tree'][1]['children'] = [3]
            skel['tree'][2]['name'] = 'RSho'
            skel['tree'][2]['children'] = [4]
            skel['tree'][3]['name'] = 'LElb'
            skel['tree'][3]['children'] = [5]
            skel['tree'][4]['name'] = 'RElb'
            skel['tree'][4]['children'] = [6]
            skel['tree'][5]['name'] = 'LWri'
            skel['tree'][5]['children'] = []
            skel['tree'][6]['name'] = 'RWri'
            skel['tree'][6]['children'] = []
            skel['tree'][7]['name'] = 'LHip'
            skel['tree'][7]['children'] = [9]
            skel['tree'][8]['name'] = 'RHip'
            skel['tree'][8]['children'] = [10]
            skel['tree'][9]['name'] = 'LKne'
            skel['tree'][9]['children'] = [11]
            skel['tree'][10]['name'] = 'RKne'
            skel['tree'][10]['children'] = [12]
            skel['tree'][11]['name'] = 'LAnk'
            skel['tree'][11]['children'] = []
            skel['tree'][12]['name'] = 'RAnk'
            skel['tree'][12]['children'] = []
            return skel

        def getPictoStruct(skel, distribution):
            """to get the pictorial structure"""
            graph = skel['tree']
            level = np.zeros ( len ( graph ) )
            for i in range ( len ( graph ) ):
                queue = np.array ( graph[i]['children'], dtype=np.int32 )
                for j in range ( queue.shape[0] ):
                    graph[queue[j]]['parent'] = i
                while queue.shape[0] != 0:
                    level[queue[0]] = level[queue[0]] + 1
                    queue = np.append ( queue, graph[queue[0]]['children'] )
                    queue = np.delete ( queue, 0 )
                    queue = np.array ( queue, dtype=np.int32 )
            trans_order = np.argsort ( -level )
            edges = [{} for i in range ( len ( trans_order ) - 1 )]
            for i in range ( len ( trans_order ) - 1 ):
                edges[i]['child'] = trans_order[i]
                edges[i]['parent'] = graph[edges[i]['child']]['parent']
                edge_id = distribution['joints2edges'][(edges[i]['child'], edges[i]['parent'])]
                edges[i]['bone_mean'] = distribution['mean'][edge_id]
                edges[i]['bone_std'] = distribution['std'][edge_id]
            return edges

        def convert_triangulation_COCO2014_to_H36m_index(points3D):
            coco2014_to_H36m = {0:  'Nose',
                                5:  'LSho',
                                6:  'RSho',
                                7:  'LElb',
                                8:  'RElb',
                                9:  'LWri',
                                10: 'RWri',
                                11: 'LHip',
                                12: 'RHip',
                                13: 'LKne',
                                14: 'RKne',
                                15: 'LAnk',
                                16: 'RAnk'}
            
            convert_points3D = []
            for i in coco2014_to_H36m.keys():
                convert_points3D.append(points3D[i])
                
            return np.array(convert_points3D)
        
        distribution = load_distribution('Unified')
        skel = getskel()
        return getPictoStruct(skel, distribution)