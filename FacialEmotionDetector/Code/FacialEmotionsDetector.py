import cv2 
from .OnnxModel import OnnxModel
import numpy as np
import os
from typing import Tuple


class FacialEmotionsDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        curr_path = os.path.dirname(os.path.abspath(__file__))
        path_to_onnx = os.path.join(curr_path[:-5], 'SavedModels', 'mobilenet_fed.onnx')
        try:
            self.onnx_model = OnnxModel(path_to_onnx)
        except Exception:
            print("Could not find onnx model's file with path {path_to_onnx}")
            raise FileNotFoundError
        self.req_img_size = (224, 224)
        self.onnx_model_img_size = (1, 1, *self.req_img_size)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5 # fontScale
        self.color = (0, 255, 0) # Green color in BGR
        self.thickness = 1 # Line thickness of 1 px
        mean = [0.508]
        std = [0.255]
        self.mean = np.array(mean).reshape(1,-1,1,1).astype(np.float32)
        self.std = np.array(std).reshape(1,-1,1,1).astype(np.float32)
        self.scale_factor = 1.3
        self.min_neights = 7
                
    def preprocess_img(self, roi_gray: np.ndarray) -> np.ndarray:
        resized_gray = cv2.resize(roi_gray, self.req_img_size, interpolation=cv2.INTER_LANCZOS4).reshape(self.onnx_model_img_size).astype(np.float32)  # default resize cv2.INTER_CUBIC
        normed_gray = resized_gray / 255.0
        standardized_gray = (normed_gray - self.mean) / self.std
        return standardized_gray

    def detect_emotions(self, source: str) -> Tuple[int, str]:
        if source == 'webcam':
            try:
                cap = cv2.VideoCapture(0)
            except Exception as e:
                print(e)
                return -1, 'could not connect to webcam'
        else:
            try:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    return -1, "Error: Could not open video_btn, maybe the file is corrupt or it is not a video"
            except Exception as e:
                return -1, 'could not open file'
        
        
        while True: 
            # reads frames from a camera
            _, img = cap.read() 
        
            # convert to gray scale of each frames
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Detects faces of different sizes in the input image
            faces = self.face_cascade.detectMultiScale(gray, self.scale_factor, self.min_neights)
        
            for (x,y,w,h) in faces:
                # To draw a rectangle in a face 
                cv2.rectangle(img, (x,y),(x+w,y+h),(0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                img_preprocessed = self.preprocess_img(roi_gray)
                probs, classes = self.onnx_model.get_img_classes_probs(img_preprocessed)
                l = len(probs)
                cls_prob_labels = map(lambda i: f"{classes[i]}: {round(probs[i], 3)}", range(l))
                char_size=20
                for i, label in enumerate(cls_prob_labels):
                    img = cv2.putText(img, label, (x, y+(i-2)*char_size), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        
            # Display an image in a window
            cv2.imshow('Facial emotions detector. Press Esc to quit', img)
        
            # Wait for Esc key to stop
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return 0, 'everything is ok'

    def detect_emotions_on_photo(self, source: str) -> Tuple[int, str]:
        img_bgr = cv2.imread(source)
        if img_bgr is None:
            return -1, 'could not open image'
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        faces = self.face_cascade.detectMultiScale(img_gray, self.scale_factor, self.min_neights)

        for (x, y, w, h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            img_preprocessed = self.preprocess_img(roi_gray)
            probs, classes = self.onnx_model.get_img_classes_probs(img_preprocessed)
            l = len(probs)
            cls_prob_labels = map(lambda i: f"{classes[i]}: {round(probs[i], 3)}", range(l))
            char_size = 20
            for i, label in enumerate(cls_prob_labels):
                img_bgr = cv2.putText(img_bgr, label, (x, y + (i - 2) * char_size),
                        self.font, self.fontScale, self.color,
                        self.thickness, cv2.LINE_AA)

            # Display an image in a window
        cv2.imshow('Facial emotions detector', img_bgr)
        return 0, 'everything is ok'