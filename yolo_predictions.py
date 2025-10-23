# Import Libraries

import cv2
import numpy as np
import pytesseract as pt
import os
import yaml
from yaml.loader import SafeLoader

# Define input width and height for the YOLO model
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

            self.labels = data_yaml['names']
            self.nc = data_yaml['nc']

            # load YOLO model
            self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def extract_text(self, img, box):
        # Extract License Plate from Image
        x, y, w, h = box
        plate_img = img[y:y+h, x:x+w]

        # Convert image to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to make text more clear
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform some operations to remove noise and make text more clear
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Apply OCR on the license plate
        text = pt.image_to_string(thresh, lang='eng', config='--psm 6')

        # Remove any special characters from the text
        text = ''.join(e for e in text if e.isalnum())

        return text


    def predictions(self, image):

        # Convert Image to YOLO Format
        img = image.copy()
        row, col, d = img.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = img

        # Get prediction from yolo model
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
        detections = preds[0]

        # Initialize empty lists for boxes and confidences
        boxes = []
        confidences = []

        # Get image width and height
        image_w, image_h = input_image.shape[:2]

        # Calculate scaling factors to convert coordinates back to original image size
        x_factor = image_w/INPUT_WIDTH
        y_factor = image_h/INPUT_HEIGHT

        # Loop through all detections and filter them based on confidence and probability score
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of detecting license plate
            if confidence > 0.1:
                class_score = row[5] # probability score of license plate
                if class_score > 0.1:
                    # Get coordinates of bounding box and scale back the orginal iamge size
                    cx, cy, w, h = row[0:4]

                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy-0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    box = np.array([left, top, width, height])

                    # Add the box and confidence to the list
                    confidences.append(confidence)
                    boxes.append(box)

        # Perform Non-Maximum Suppression to remove duplicate detections
        index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

        # Create a copy of the input image for output
        output_image = input_image.copy()

        for ind in index:
            # Extract coordinates and confidence of the bounding box
            x, y, w, h = boxes[ind]
            bb_conf = confidences[ind]

            # Create text to show the confidence of the bounding box
            conf_text = 'Plate:{:.0f}%'.format(bb_conf*100)

            # Extract text from the license plate using OCR
            license_text = self.extract_text(input_image, boxes[ind])

            # Draw rectangles for the bounding box, confidence text
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(output_image,(x,y-30),(x+w,y),(255,0,0),-1)
            cv2.rectangle(output_image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)
            
            cv2.putText(output_image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(output_image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)



        # Crop the output image to the actual size of the input image
        #output_image = output_image[0:row, 0:col]

        return output_image, license_text



