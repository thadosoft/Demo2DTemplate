import onnxruntime
import torch
import numpy as np

from ultralytics import YOLO

class ULObjectDetection:
    def __init__(self, ckpt_path, model_name):
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.model = YOLO(f'{ckpt_path}\\{model_name}.pt')
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.tensorrt_model = None
    
    def export_tensorrt(self, fresh):
        if fresh is True:
            self.model.export(format="engine")
        self.tensorrt_model = YOLO(f'{self.ckpt_path}\\{self.model_name}.engine', task='detect')

    def predict(self, mode, img):
        if mode == 'normal':
            return self.model(img, conf=0.2)
        elif mode == 'tensorrt':
            return self.tensorrt_model.predict(img, conf=0.3, iou=0.2, verbose=False, device=0)
