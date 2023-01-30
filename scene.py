import numpy as np
import time
import pyrealsense2 as rs
from functools import wraps
import main_algorithm
import cv2
from buffer import buffer
import copy
import matplotlib.pyplot as plt
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


class scene():
    def __init__(self, 
                cameras: list,
                pplDetect_model,
                pose2D_model,
                pictoStruct: list,
                merge_singleView_width: int,
                merge_singleView_height: int,
                merge_col_num: int,
                cam_idx_pair: list,
                ):

        self.pplDetect_model = pplDetect_model
        self.pose2D_model = pose2D_model
        self.pictoStruct = pictoStruct
        self.merge_singleView_width = merge_singleView_width
        self.merge_singleView_height = merge_singleView_height
        self.merge_col_num = merge_col_num
        self.cameras = cameras
        self.cam_idx_pair = cam_idx_pair

        self.cam_nums = len(cameras)
        self.frame_num = 0

        self.fundamentalMatPts = [np.zeros((1, 2), dtype=np.int32) for _ in range(len(cameras))]
        # self.fundamentalMatPts1 = buffer(100)
        # self.fundamentalMatPts2 = buffer(100)

    def get_show_actors(self) -> list:

        actors = []
        self.frame_num += 1
        self.show_images = self.get_show_images(self.cameras[0].camera_type)
        print('---------------------------------------')
        print(f'Frame num: {self.frame_num}')
        
        # Stage1, get id matching and 2D pose (pixels)
        # match_list, detect_info = self.get_match_list(image_nparray)
        match_list, self.show_images, detect_info = main_algorithm.get_match_pose2D(self.show_images,
                                                                                self.pplDetect_model,
                                                                                self.pose2D_model,
                                                                                self.cam_nums,
                                                                                self.merge_singleView_width,
                                                                                self.merge_singleView_height,
                                                                                self.merge_col_num)
        if match_list:
            actors = main_algorithm.reconstruction_pose3D(self.cameras,
                                                        self.cam_idx_pair,
                                                        self.pictoStruct,
                                                        match_list,
                                                        detect_info)

            # res = self.recover3Dpose(image_nparray, match_list, detect_info)

        return actors


    def get_show_images(self, camera_type) -> list:
        # No matter which type of camera, should have a list which collect frames with nparray type.
        image_nparray = []

        match camera_type:
            case 'realsense':
                # wait for data
                frames = []
                align = rs.align(rs.stream.color)
                for camera in self.cameras:
                    frames.append(camera.read())

                # align the input frame
                aligned_frames = []
                for f in frames:
                    aligned_frames.append(align.process(f))

                # grab those aligned frame
                aligned_color_frames = []
                for a_f in aligned_frames:
                    aligned_color_frames.append(a_f.get_color_frame())
    
                for a_c_f in aligned_color_frames:
                    image_nparray.append(np.asanyarray(a_c_f.get_data()))

            case 'video':
                for camera in self.cameras:
                    image_nparray.append(camera.read())


            
            case 'ipcam':
                for camera in self.cameras:
                    image_nparray.append(camera.read())

        return image_nparray

    def findFundamentalMat(self):

        def drawlines(img1,img2,lines,pts1,pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r,c, _ = img1.shape
            # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
            # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            for r,pt1,pt2 in zip(lines,pts1,pts2):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
                img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
                img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
            return img1,img2

        self.frame_num += 1
        self.show_images = self.get_show_images(self.cameras[0].camera_type)
        print('---------------------------------------')
        print(f'Frame num: {self.frame_num}')
        
        # Stage1, get id matching and 2D pose (pixels)
        # match_list, detect_info = self.get_match_list(image_nparray)
        match_list, self.show_images, detect_info = main_algorithm.get_match_pose2D(self.show_images,
                                                                                self.pplDetect_model,
                                                                                self.pose2D_model,
                                                                                self.cam_nums,
                                                                                self.merge_singleView_width,
                                                                                self.merge_singleView_height,
                                                                                self.merge_col_num)

        

        print(f'Info Lens: {len(detect_info)}')
        for v1, v2 in self.cam_idx_pair:
            # print(len(detect_info[v1]))
            # print(len(detect_info[v2]))
            if detect_info[v1] and detect_info[v2]:
                # Only one person, so idx fixed as 0
                print(f'CAM: {v1} and {v2}')
                for i in range(detect_info[v1][0]['keypoints'].shape[0]):
                    if detect_info[v1][0]['keypoints'][i][2] > 0.7 and detect_info[v2][0]['keypoints'][i][2] > 0.7:
                        self.fundamentalMatPts[v1] = np.concatenate((self.fundamentalMatPts[v1], detect_info[v1][0]['keypoints'][i, :2].reshape((1, 2)).astype('int32')), axis = 0)
                        self.fundamentalMatPts[v2] = np.concatenate((self.fundamentalMatPts[v2], detect_info[v2][0]['keypoints'][i, :2].reshape((1, 2)).astype('int32')), axis = 0)
                # self.fundamentalMatPts[v1].push(detect_info[v1][0]['keypoints'][:, :2])
                # self.fundamentalMatPts[v2].push(detect_info[v2][0]['keypoints'][:, :2])


                pts1 = self.fundamentalMatPts[v1][1:, :].copy()
                pts2 = self.fundamentalMatPts[v2][1:, :].copy()
                if pts1.shape[0] > 8:
                    # print(f'pts1: {pts1}')
                    # print(f'pts2: {pts2}')
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
                    print(F)
                    # pts1 = pts1[mask.ravel()==1]
                    # pts2 = pts2[mask.ravel()==1]
                    # Find epilines corresponding to points in right image (second image) and
                    # drawing its lines on left image
                    if pts1.shape[0] > 20:
                        n = 19  # for 2 random indices
                        index = np.random.choice(pts1.shape[0], n, replace=False)  
                        pts1 = pts1[index]
                        pts2 = pts2[index]
                    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
                    lines1 = lines1.reshape(-1,3)
                    img5,img6 = drawlines(self.show_images[v1], self.show_images[v2],lines1,pts1,pts2)
                    # Find epilines corresponding to points in left image (first image) and
                    # drawing its lines on right image
                    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
                    lines2 = lines2.reshape(-1,3)
                    img3,img4 = drawlines(self.show_images[v2], self.show_images[v1],lines2,pts2,pts1)
                    cv2.imshow(f'cam{v1}{v2}', np.concatenate((img5, img3), axis=1))
                    key = cv2.waitKey(1)

                    if key == 27:
                        break
                


