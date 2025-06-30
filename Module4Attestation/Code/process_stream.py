import cv2 
from .OnnxModel import OnnxModel
import numpy as np
import os


def process_stream(source: str):
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video, maybe the file is corrupt or it is not a video")
            exit()
            return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path_to_onnx = os.path.join(curr_path[:-5], 'SavedModels', 'resnet18_fl.onnx')
    try:
        onnx_model = OnnxModel(path_to_onnx)
    except Exception:
        print("Could not find onnx model's file with path {path_to_onnx}")
        return
    
    
    req_img_size = (96, 96)
    resnet18_model_img_size = (1, 1, 96, 96)
    
    color = (0, 255, 0) # Green color
    radius = 2
    thickness = -1
    
    while True: 
        # reads frames from a camera
        _, img = cap.read() 
    
        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            # To draw a rectangle in a face 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
            roi_gray = gray[y:y+h, x:x+w]
            coef_h = h / req_img_size[0]
            coef_w = w / req_img_size[1]
            resized_gray = cv2.resize(roi_gray, req_img_size, interpolation=cv2.INTER_CUBIC).reshape(resnet18_model_img_size).astype(np.float32)
            normed_gray = resized_gray / 255.0
            points = onnx_model(normed_gray)[0]
            for point in points:
                coords = [int(x+point[0]*coef_w), int(y+point[1]*coef_h)]
                cv2.circle(img, coords, radius, color, thickness)
    
        # Display an image in a window
        cv2.imshow('Facial keypoints detector. Press Esc to quit', img)
    
        # Wait for Esc key to stop
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()