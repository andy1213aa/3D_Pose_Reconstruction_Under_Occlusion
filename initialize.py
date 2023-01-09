

def scene_initialize(self):

    self.init_cameras() 
    self.init_model()
    self.pictoStruct = pictorial_3D.init_3DPS()
    self.init_scene3D()
    self.cam_nums = len(self.cameras)

    