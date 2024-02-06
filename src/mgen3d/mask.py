
import cv2
import numpy as np


class MaskExtractor:
    def __init__(self):
        pass

    def extract(self, image):
        print(image.shape)
        assert image.shape[2] == 4, "Image must have 4 channels, RGBA."
        # extract alpha channel
        alpha = image[:, :, 3]

        # threshold alpha channel
        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        return mask

