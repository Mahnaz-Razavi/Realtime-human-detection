"""
config.py is written to save configurations for human detection project
"""

import argparse

__authors__ = "Mahnaz Razavi"
__project__ = "Human Detection"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Mahnaz Razavi"
__email__ = "mah.razavi90@gmail.com"
__status__ = "Production"


class Config:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statistics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        """
        run module to start definition of configurations

        """
        # Required parameters
        self.parser.add_argument('--input',
                                 default='stream', help='image', type=str)
        self.parser.add_argument('--stream_source',
                                 default='video', help='camera' or 'webcam', type=str)
        self.parser.add_argument('--dir_images',
                                 default='../data/inputs/images/')
        self.parser.add_argument('--dir_video',
                                 default='../data/inputs/video_test/test.mp4')
        self.parser.add_argument('--rtsp_camera',
                                 default='rtsp://')
        self.parser.add_argument('--port_webcam',
                                 default='0')
        self.parser.add_argument('--output_stream',
                                 default='../data/outputs/output.avi')
        self.parser.add_argument('--output_image',
                                 default='../data/outputs/image/')

        # MobileNet config
        self.parser.add_argument('--detection_size', default=(300, 300))
        self.parser.add_argument('--scale_factor', default=0.007843)
        self.parser.add_argument('--mean', default=127.5)
        self.parser.add_argument('--fps', default=25, type=int)
        self.parser.add_argument('--det_thresh', default=0.01, type=float)
        self.parser.add_argument('--txt_dir',
                                 default='../models/pre_trained/human_detection/MobileNet_SSD/MobileNetSSD_deploy.prototxt',
                                 type=str)
        self.parser.add_argument('--model_dir',
                                 default='../models/pre_trained/human_detection/MobileNet_SSD/MobileNetSSD_deploy.caffemodel',
                                 type=str)

        # frame shape
        self.parser.add_argument('--frame_height', default=480, type=int)
        self.parser.add_argument('--frame_width', default=640, type=int)

    def get_args(self):
        """
        get_args module to return defined configurations

        """
        return self.parser.parse_args()
