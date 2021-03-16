import cv2
import dlib
import numpy as np 
from imutils import face_utils
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--video", default='0', help="video path")
args = vars(ap.parse_args())
video_path = args["video"]
if video_path == '0':
    video_path = 0

detector = dlib.get_frontal_face_detector()
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)

cam = cv2.VideoCapture(video_path)
ret_val, frame = cam.read()

while True:
    ret_val, frame = cam.read()
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
        landmarks = predictor(image, rect)
        landmarks = face_utils.shape_to_np(landmarks)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)
        
        for i in range(len(landmarks)):
            (x,y) = landmarks[i]
            cv2.circle(image, (x, y), 3, (0,0,255), -1)
            if i < len(landmarks)-1:
                (x2, y2) = landmarks[i+1]
                cv2.line(image, (x, y), (x2, y2), (0,255,255), 1)
            
    
    cv2.imshow('Output', image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

