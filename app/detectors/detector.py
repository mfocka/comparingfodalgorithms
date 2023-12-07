from abc import abstractmethod
from pathlib import Path
import cv2

import numpy as np

try:
    # Import for when using Linux
    import tflite_runtime.interpreter as tflite
except:
    # Import for when using Windows
    import tensorflow.lite as tflite


class Detector:
    def __init__(self, modelName, modelLocation, labelLocation, modelExtension, confidenceThreshold, onEdge, doTpu):
        self.name = modelName
        self.model    = self.get_model(f'{modelLocation}model{modelExtension}')
        self.labelMap = self.get_label_map(f'{labelLocation}labelMap.txt')
        self.confidenceThreshold = confidenceThreshold
        self.onEdge = onEdge
        if not onEdge:
            self.doTPU = doTpu
        self.interpreter = None
    
    def get_model(self, modelLocation):
        """
        Load the model file and set the modelLocation attribute.
        """
        if not Path(modelLocation).exists():
            self.modelLocation = str((Path(__file__).parent / Path(
                modelLocation)).resolve().absolute())
        else:
            self.modelLocation = modelLocation
        return self.modelLocation

    def get_label_map(self, labelLocation):
        """
        Load the labelmap file and set the labels attribute.
        """
        if not Path(labelLocation).exists():
            self.labelLocation = str((Path(__file__).parent / Path(
                labelLocation)).resolve().absolute())
        else:
            self.labelLocation = labelLocation
        labelNames = [line.rstrip('\n') for line in open(self.labelLocation)]
        return np.array(labelNames)
    
    def parse_detection_tensors(self):
        detection_boxes, detection_scores, detection_classes, num_detections = self.get_output_detection_groups()
        
        detections = []
        for detection_id in range(num_detections):
            if detection_scores[detection_id] > self.confidenceThreshold:
                y_start, x_start, y_end, x_end = detection_boxes[detection_id]
                class_id = int(detection_classes[detection_id])
                try:
                    class_name = self.labelMap[class_id]
                except:
                    class_name = self.labelMap[0]
                conf = detection_scores[detection_id]
                detections.append({
                    "name": class_name,
                    "class_id": class_id,
                    "bounding_box": (x_start, y_start, x_end, y_end),
                    "confidence": conf
                })
        return detections
    
    def get_detections(self, frame):
        if self.interpreter is None:
            self.setup_interpreter()
        nFrame = self.prepare_input(frame)
        self.perform_inference(nFrame)
        return self.parse_detection_tensors()
    
    def prepare_input_on_embedded(self, frame):
        assert self.inputDetails is not None
        assert len(frame.shape) >= 3
        d, w, h, c = self.inputDetails[0]['shape']
        self.model_size = (w,h)
        disp_image = cv2.resize(frame, (w,h))
        image = (disp_image - 0) / 255
        image = (image * 255).astype(self.inputDetails[0]['dtype'])
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return image
    def prepare_input(self, frame):
        self.frameSize = frame.shape
        if not self.onEdge:
            return self.prepare_input_on_embedded(frame)
        return frame
            
    
    def draw_detections(self, frame, detections):
        if len(detections) > 0:
            COLORS = np.random.randint(0, 255, size=(
                int(max([detection['class_id'] for detection in detections])+1), 3), dtype=np.uint8)
            for detection in detections:
                bbox = detection["bounding_box"]
                color = [int(c) for c in COLORS[detection['class_id']]]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                label = "{}: {:.0f}%".format(detection['name'], detection['confidence'] * 100)
                cv2.putText(frame, label, (bbox[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def perform_inference_on_embedded(self, inData):
        self.interpreter.set_tensor(self.get_input_index(), inData)
        self.interpreter.invoke()
    def setup_interpreter_on_embedded(self):
        """Initialize the TFLite interpreter."""
        ext_delegate = []
        if self.doTPU:
            external_delegate_path = '/usr/lib/libvx_delegate.so'
            ext_delegate_options = {
                # 'device': 'NPU',
                # 'target': 'imx8mplus'
            }
            print('Loading external delegate from {} with args {}'.format(
                external_delegate_path, ext_delegate_options))
            ext_delegate = [
                tflite.load_delegate(
                    external_delegate_path, ext_delegate_options)
            ]

        self.interpreter = tflite.Interpreter(
            model_path=self.modelLocation,
            experimental_delegates=ext_delegate)
        self.interpreter.allocate_tensors()

        self.inputDetails = self.interpreter.get_input_details()
        self.outputDetails = self.interpreter.get_output_details()
    def warm_up_inference_on_embedded(self):
        assert self.interpreter is not None
        self.interpreter.invoke()
        
    def nms(self, detections, overlapThresh=0.4):
        # Return an empty list, if no boxes given

        boxes_per_class = {}
        for detection in detections:
            boxes_per_class.setdefault(
                detection['class_id'], []).append(detection)
        final_detections = []
        for class_id, class_detections in boxes_per_class.items():
            sorted_class_detections = sorted(
                class_detections, key=lambda item: item['confidence'], reverse=True)
            boxes = np.array([list(detection['bounding_box'])
                              for detection in sorted_class_detections])

            x1 = boxes[:, 1]  # x coordinate of the top-left corner
            y1 = boxes[:, 0]  # y coordinate of the top-left corner
            x2 = boxes[:, 3]  # x coordinate of the bottom-right corner
            y2 = boxes[:, 2]  # y coordinate of the bottom-right corner
            # Compute the area of the bounding boxes and sort the bounding
            # Boxes by the bottom-right y-coordinate of the bounding box
            # We add 1, because the pixel at the start as well as at the end counts
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            # The indices of all boxes at start. We will redundant indices one by one.
            indices = np.arange(len(x1))
            for i, box in enumerate(boxes):
                # Create temporary indices
                temp_indices = indices[indices != i]
                # Find out the coordinates of the intersection box
                xx1 = np.maximum(box[0], boxes[temp_indices, 0])
                yy1 = np.maximum(box[1], boxes[temp_indices, 1])
                xx2 = np.minimum(box[2], boxes[temp_indices, 2])
                yy2 = np.minimum(box[3], boxes[temp_indices, 3])
                # Find out the width and the height of the intersection box
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                # compute the ratio of overlap
                overlap = (w * h) / areas[temp_indices]
                # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
                if np.any(overlap) > overlapThresh:
                    indices = indices[indices != i]
            # return only the boxes at the remaining indices
            # return boxes[indices].astype(int)
            final_detections += [sorted_class_detections[i] for i in indices]
        return final_detections
    
    @abstractmethod
    def perform_inference(self, frame):
        if not self.onEdge:
            self.perform_inference_on_embedded(frame)
        else:
            raise NotImplementedError
    
    @abstractmethod
    def warm_up_inference(self):
        if not self.onEdge:
            self.warm_up_inference_on_embedded()
        else:
            raise NotImplementedError
    
    @abstractmethod
    def get_output_detection_groups(self):
        raise NotImplementedError
        
    @abstractmethod
    def setup_interpreter(self):
        if not self.onEdge:
            self.setup_interpreter_on_embedded()
        else:
            raise NotImplementedError
        
    @abstractmethod
    def get_input_index(self):
        if not self.onEdge:
            raise NotImplementedError
    