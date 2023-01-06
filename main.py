from scene import scene
import config

if __name__ == '__main__':
    show = scene(config.video_infomation)
    show.scene_initialize()
    show.start()