import numpy as np
from detectors.detector import Detector


class YOLOv8(Detector):
  def __init__(self, modelLocation, labelLocation, modelExtension, confidenceThreshold, onEdge, doTpu):
      super().__init__("YOLOv8s", f'{modelLocation}/yolov8s/', f'{labelLocation}/yolov8s/',modelExtension, confidenceThreshold, onEdge,doTpu)

  def get_input_index(self):
      return self.inputDetails[0]['index']
  def get_output_index(self):
      return self.outputDetails[0]['index']
  def get_output_detection_groups(self):
    assert self.outputDetails is not None
    output = self.interpreter.get_tensor(self.get_output_index())
    output = output[0]
    output = np.transpose(output.reshape(84, 8400))

    detections = []
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    _, mw, mh, _ = self.inputDetails[0]['shape']
    for row in output:
        xc,yc,w,h = row[:4]
        x1 = (xc-w/2) *mw
        y1 = (yc-h/2) *mh
        x2 = (xc+w/2) *mw
        y2 = (yc+h/2) *mh
        prob = row[4:].max()
        class_id = int(row[4:].argmax())
        detections.append({
                        "class_id": class_id,
                        "bounding_box": (y1,x1,y2,x2),
                        "confidence": prob
                    })
    detections_list = self.nms(detections)
    detection_boxes = []
    for box in [d['bounding_box'] for d in detections_list]:
        detection_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    
    detection_scores = [d['confidence'] for d in detections_list]
    detection_classes = [d['class_id'] for d in detections_list]
    return detection_boxes, detection_scores, detection_classes, len(detection_classes)


