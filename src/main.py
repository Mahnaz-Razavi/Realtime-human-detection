"""
main.py is written to run the model under deployment
"""

from config import Config
import cv2
from human_detection.MobileNet_SSD import HumanDetection
from utils import PostProcess, PreProcess

__author__ = "Mahnaz Razavi"
__project__ = "Human Detection"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Mahnaz Razavi"
__email__ = "mah.razavi90@gmail.com"
__status__ = "Production"


class Run(object):
    """
    Run class
    get all modules together and create a sensible procedure
    """

    def __init__(self, args):
        """
        Initialize modules
        """
        super(Run, self).__init__()

        self.args = args

        self.obj_preprocess = PreProcess(self.args)
        self.obj_detection = HumanDetection(self.args)
        self.obj_postprocess = PostProcess()

    def start(self):
        """
        start module
        To start human detection process

        """
        if self.args.input == 'stream':
            out = self.obj_preprocess.create_video_writer()

            while 1:
                frame = self.obj_preprocess.read_frame()

                if frame is None:
                    self.obj_preprocess.recapture()
                    continue

                frame = cv2.resize(frame, (self.args.frame_width, self.args.frame_height))

                human_boxes, accuracies = self.obj_detection.human_detection(frame)

                frame = self.obj_postprocess.plt_box(frame, human_boxes, accuracies)

                out.write(frame)
                

if __name__ == '__main__':
    args = Config().get_args()
    obj_run = Run(args)
    obj_run.start()
