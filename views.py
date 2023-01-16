import numpy as np
import cv2
import utlis


class Visualization():
    def __init__(self,show_FPS,
                show_kpt2D,
                show_scene2D,
                show_scene3D,
                ax,
                merge_col_num,
                merge_singleView_height,
                merge_singleView_width,):
        
        

        self.show_FPS = show_FPS
        self.show_kpt2D = show_kpt2D
        
        self.show_scene2D = show_scene2D
        self.show_scene3D = show_scene3D
        
    
        self.visWindow2D = cv2.namedWindow('view2D',cv2.WINDOW_NORMAL)
        self.visWindow3D = cv2.namedWindow('view3D',cv2.WINDOW_NORMAL)

        self.ax = ax

        self.merge_col_num = merge_col_num
        self.merge_singleView_height = merge_singleView_height
        self.merge_singleView_width = merge_singleView_width
        
        self._show_images = np.zeros((merge_singleView_height, merge_singleView_width))

    def result_visualization(self, 
                        exec_time='No Given'):
        while True:

            if self.show_scene2D:
                view2D = self._vis2D(self._show_images, 
                                    exec_time)
                cv2.imshow('view2D', view2D)

            if self.show_scene3D:
                scene3D = self._vis3D()
                cv2.imshow('view3D', scene3D)
                
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
   
                break

            elif key == ord('f') or key == ord('F'):
                self.show_FPS = not self.show_FPS

            elif key == ord('k') or key == ord('K'):
                self.show_kpt2D = not self.show_kpt2D
                self.pose2D.visualization = self.show_kpt2D

            elif key == ord('v') or key == ord('V'):
                self.show_scene3D = not self.show_scene3D
                if not self.show_scene3D:
                    cv2.destroyWindow('view3D')

            # rotation
            elif key == ord('w') or key == ord('W'):
                self.ax.elev -= 1
            elif key == ord('a') or key == ord('A'):
                self.ax.azim -= 1
            elif key == ord('s') or key == ord('S'):
                self.ax.elev += 1
            elif key == ord('d') or key == ord('D'):
                self.ax.azim += 1        



    def _vis2D(self, 
            show_images, 
            exec_time
            ):

        for i, view in enumerate(show_images):
            
            if self.show_FPS:
                fps = int(1 / (exec_time))
                cv2.putText(view, f'FPS: {fps}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            self.cameras[i].now_frame = view
        
        merge_multi_view = utlis.merge_images(self._show_images, self.merge_col_num, self.merge_singleView_height, self.merge_singleView_width)

        # if record_result:
        #     MV.write(merge_multi_view)
        # if self.record_result:
        #             D3V.write(scene3D)
        return merge_multi_view
            
    def _vis3D(self):

        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]
    # 3D
    #         skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
    #                 [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
    #                 [8, 14], [14, 15], [15, 16]]
        # 棋盤格
        width = 7
        height = 5
        objp = np.zeros((height*width,3), np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
        objp *= 6.8

        hd_cameras = [self.cam_info(self.cameras[i].calibration_parameter['R'], 
                        self.cameras[i].calibration_parameter['t']) for i in range(len(self.cameras))]

        if self.cfg['world_view']:
            # Draw selected camera subset in blue   
            for cam in hd_cameras:
                cc = -cam['R'].transpose()@cam['t']
                self.ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,0,1], s=10)
                
            # Draw chessboard
            for objpt in objp:
                self.ax.scatter(objpt[0], objpt[1], objpt[2], '.', color=[1,0,0], s= 1)

        if self.actors:

            for ppl in self.actors:
    #             kpt3d_world = self.convert_keypoint_definition(ppl.win_kpts3D)
                
                # Denormalize
                kpt3d_world = ppl.win_kpts3D * self.cfg['cam_denormalize'][ppl.win_cam_pair[0]]
                # transformation to world coordination
                if self.cfg['world_view']:
                    kpt3d_world = self.convert_keypoint_cam2world(kpt3d_world, ppl.win_cam_pair)

                for objpt in kpt3d_world:
                    self.ax.scatter(objpt[0], objpt[1], objpt[2], '.', color=[1,0,0], s=0.5)

                for skl in skeleton:
                    self.ax.plot([kpt3d_world[skl[0]][0], kpt3d_world[skl[1]][0]], 
                        [kpt3d_world[skl[0]][1], kpt3d_world[skl[1]][1]], 
                        [kpt3d_world[skl[0]][2], kpt3d_world[skl[1]][2]])

        self.canvas.draw()       # draw the canvas, cache the renderer
        scene_pltshowing = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(self.canvas_height), 
                                                                                            int(self.canvas_width), 3) 

        # cv2.imwrite(f'vis/test_{self.frame_num}.png', scene_pltshowing)
        self.ax.cla()
        return scene_pltshowing

    def update_show_images(self, new_show_images):
        self._show_images = new_show_images

    def convert_keypoint_definition(self, keypoints):
        """Convert pose det dataset keypoints definition to pose lifter dataset
        keypoints definition.

        Args:
            keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
            pose_det_dataset, (str): Name of the dataset for 2D pose detector.
            pose_lift_dataset (str): Name of the dataset for pose lifter model.
        """
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new

    def convert_keypoint_cam2world(self, keypoints, cams_idx):

        keypoints = keypoints.reshape(17, 3, 1)
        R_W = self.cameras[cams_idx[0]].calibration_parameter['R']
        t_W = self.cameras[cams_idx[0]].calibration_parameter['t']
        world_kpt3d = np.zeros((keypoints.shape[0], 3, 1))
        for i, kpt3D in enumerate(keypoints):
            world_kpt3d[i] =R_W.T@(kpt3D - t_W)

        world_kpt3d[:, 2] -= world_kpt3d[-1, 2]
        # 先寫死
        world_kpt3d = world_kpt3d.reshape((world_kpt3d.shape[0], 3))

        if cams_idx[0] == 2:
            RR = np.array([[    0.98472,   -0.073716,     0.15775],
                            [   0.066247,     0.99644,    0.052097],
                            [   -0.16103,   -0.040851,      0.9861]])
                            
            tt = np. array([[    -36.759],
                            [    -6.3854],
                            [    -7.7052]])

            world_kpt3d = (RR@world_kpt3d.T).T + tt.reshape(3, )

        return world_kpt3d

