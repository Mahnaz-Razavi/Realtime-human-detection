"""
postprocess.py script is written to draw boxes on each frame
"""

import cv2


class PostProcess:
    """
    PostProcess class
    run post process modules
    """
    def __init__(self) -> None:
        super(PostProcess, self).__init__()

    def plt_box(self, image, boxes, confs):
        """
        plt_box module is written to return image with bounding boxes of human
        :param:
            image: cv2 numpy.ndarray
            boxes: list of positions
            confs: list of accuracies
        :return:
            image: cv2 numpy.ndarray
        """
        for idx in range(len(boxes)):
            cv2.rectangle(image, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (0, 255, 0), 2)
            cv2.putText(image, str(confs[idx]), (boxes[idx][0], boxes[idx][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        return image
