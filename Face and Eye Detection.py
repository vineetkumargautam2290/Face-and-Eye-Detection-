import os
import numpy as np
import pandas as pd
import cv2

# face detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# eye detection
eye_classifer = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('multiple-portraits-of-young-mans-face-EN7YKX.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(image, 1.3, 5)

if faces is ():
    print("No faces found")

    # Boundtry box
for (x,y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h),(0,0,255),2)
    cv2.imshow("Face Detection", image)
    cv2.waitKey()

    #eye_image = image[y:y+h, x:x+w]
    eye_color = image[y:y+h, x:x+w]
    eyes = eye_classifer.detectMultiScale(eye_color)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (255,0,0),3)
        cv2.imshow('eye', image)
        cv2.waitKey()

cv2.destroyAllWindows()