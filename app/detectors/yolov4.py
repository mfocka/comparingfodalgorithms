from detectors.detector import Detector

class YOLOv4(Detector):
    def __init__(self, modelLocation, labelLocation, modelExtension, confidenceThreshold, onEdge, doTpu):
        super().__init__("YOLOv4", f'{modelLocation}/yolov4/', f'{labelLocation}/yolov4/', modelExtension, confidenceThreshold, onEdge, doTpu)
        
    def get_output_detection_groups(self):
        assert self.outputDetails is not None
        
        # Print output details for inspection
        print("Output Details:")
        for output_detail in self.outputDetails:
            print(output_detail)

        t_detection_boxes = self.interpreter.get_tensor(self.outputDetails[0]['index'])[0]
        detection_scores = self.interpreter.get_tensor(self.outputDetails[1]['index'])[0]
        detection_classes = self.interpreter.get_tensor(self.outputDetails[2]['index'])[0]

        detection_boxes = []
        for obj in t_detection_boxes:
            ymin, xmin, ymax, xmax = obj
            xmin = int(xmin * self.frameSize[1])
            xmax = int(xmax * self.frameSize[1])
            ymin = int(ymin * self.frameSize[0])
            ymax = int(ymax * self.frameSize[0])
            detection_boxes.append([ymin, xmin, ymax, xmax])
            
        num_detections = len(detection_classes)
        return detection_boxes, detection_scores, detection_classes, num_detections
    
    
    def get_input_index(self):
        return self.inputDetails[0]['index']
