import cv2
import numpy as np
from copy import deepcopy
import supervision as sv

from src.UL.ULObjectDetection import ULObjectDetection
from src.utils.funcs import *

class Monitor:
    def __init__(self) -> None:
        # Anomaly
        self.model = None

    def Detect(self, img):
        # Write your code here

        return img
