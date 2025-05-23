import openvino as ov
from anomalib.deploy import OpenVINOInferencer

class AnomalyDetection:
    def __init__(self, ckptPath) -> None:
        path = f'{ckptPath}/model.bin' 
        metadata = f'{ckptPath}/metadata.json'

        self.inferencer = OpenVINOInferencer(
                path = path,
                metadata = metadata,
                device = 'CPU'
                )

    def predict(self, img):
        predictions = self.inferencer.predict(img)
        print(predictions.pred_score, predictions.pred_label)
        return predictions.heat_map, predictions.segmentations, predictions.anomaly_map
