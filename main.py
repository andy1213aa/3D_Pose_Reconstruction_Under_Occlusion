from scene import scene
from config import Config
from controller import StageManager

if __name__ == '__main__':
    configure = Config()

    SM = StageManager(configure)
    SM.initlize()
    SM.start()