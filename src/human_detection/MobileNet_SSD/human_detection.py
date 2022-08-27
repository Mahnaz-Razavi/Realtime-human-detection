"""
human_detection.py script is written to detect humans in frames
"""
import cv2
import numpy as np

__author__ = "Mahnaz Razavi"
__project__ = "Human Detection"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Mahnaz Razavi"
__email__ = "mah.razavi90@gmail.com"
__status__ = "Production"


class HumanDetection:
    """
    HumanDetection class

    """
    def __init__(self, args) -> None:
        super(HumanDetection, self).__init__()
        self.args = args
        prototxt = self.args.txt_dir
        model = self.args.model_dir
        # load our serialized model from disk
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def human_detection(self, image):
        """
        human_detection module is written to return human positions and accuracies
        :param:
            image: cv2 numpy.ndarray
        :return:
            boxes: list of positions
            confidences: list of accuracies
        """
        boxes = []
        confidences = []

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, self.args.detection_size),
                                     self.args.scale_factor, self.args.detection_size,
                                     self.args.mean)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        if detections.any():
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.args.det_thresh:
                    # extract the index of the class label from the
                    # `detections`
                    idx = int(detections[0, 0, i, 1])

                    # check the predicted label is person
                    if self.CLASSES[idx] == 'person':
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        box[box < 0] = 0
                        boxes.append(box.astype("int"))
                        confidences.append(np.round(confidence, 2))

        return boxes, confidences
