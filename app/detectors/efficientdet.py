from detectors.detector import Detector


class EfficientDet(Detector):
    def __init__(self, modelLocation, labelLocation, modelExtension, confidenceThreshold, onEdge, doTpu):
        super().__init__("EfficientDet", f'{modelLocation}/efficientdet/', f'{labelLocation}/efficientdet/',modelExtension, confidenceThreshold, onEdge,doTpu)
    def get_output_detection_groups(self):
        assert self.outputDetails is not None
        
        t_detection_boxes = [[val for val in box] for box in self.interpreter.get_tensor(
            self.outputDetails[0]['index'])[0]]
        detection_boxes = []
        for obj in t_detection_boxes:
            ymin, xmin, ymax, xmax = obj
            xmin = int(xmin * self.frameSize[1])
            xmax = int(xmax * self.frameSize[1])
            ymin = int(ymin * self.frameSize[0])
            ymax = int(ymax * self.frameSize[0])
            detection_boxes.append([ymin,xmin,ymax,xmax])
            
        detection_scores = self.interpreter.get_tensor(
            self.outputDetails[2]['index'])[0]
        detection_classes = self.interpreter.get_tensor(
            self.outputDetails[1]['index'])[0]
        num_detections = len(detection_boxes)


        return detection_boxes, detection_scores, detection_classes, num_detections
    
    def get_input_index(self):
        return self.inputDetails[0]['index']