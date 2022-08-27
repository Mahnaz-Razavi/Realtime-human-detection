"""
preprocess.py script is written to prepare frame
"""

import cv2
from vidgear.gears import CamGear

__author__ = "Mahnaz Razavi"
__project__ = "Human Detection"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Mahnaz Razavi"
__email__ = "mah.razavi90@gmail.com"
__status__ = "Production"


class PreProcess:
    """
    PreProcess class
    run pre process modules
    """
    def __init__(self, args) -> None:
        super(PreProcess, self).__init__()

        self.args = args

        if self.args.input == 'stream':
            if self.args.stream_source == 'webcam':
                self.cap = self.cap_from_webcam()
            if self.args.stream_source == 'video':
                self.cap = self.cap_from_video()
            if self.args.stream_source == 'camera':
                self.cap = self.cap_from_camera()

    def cap_from_webcam(self):
        """
        capture frames from webcam
        :return:
            cap: videocapture
        """
        cap = cv2.VideoCapture(int(self.args.port_webcam))
        print('port webcam :', str(self.args.port_webcam))
        return cap

    def cap_from_video(self):
        """
        capture frames from video
        :return:
            cap: videocapture
        """
        cap = cv2.VideoCapture(self.args.dir_video)
        print('directory video :', str(self.args.dir_video))
        return cap

    def cap_from_camera(self):
        """
        capture frames from camera
        :return:
            cap: videocapture
        """
        options = {'THREADED_QUEUE_MODE': False}
        stream = CamGear(source=self.args.rtsp_camera, **options).start()
        print('rtsp camera : ',str(self.args.rtsp_camera))
        return stream

    def read_frame(self):
        """
        read frame from VideoCapture
        :return:
            frame: cv2 numpy.ndarray
        """
        # mode frame read
        if self.args.stream_source == 'video' or self.args.stream_source == 'webcam':
            ret, frame = self.cap.read()
        elif self.args.stream_source == 'camera':
            frame = self.cap.read()

        return frame

    def recapture(self):
        """
        capture again when camera disconnected and then reconnected
        :return:
            cap: videocapture
        """
        print('Frame is None')
        if self.args.stream_source == 'camera':
            self.cap = self.cap_from_camera()

    def create_video_writer(self):
        """
        create_video_writer module is written to create VideoWriter
        :return:
            out: VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            self.args.output_stream,
            fourcc=fourcc,
            fps=self.args.fps,
            frameSize=(self.args.frame_width, self.args.frame_height)
        )

        return out
