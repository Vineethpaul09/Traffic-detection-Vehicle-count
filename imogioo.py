import cv2
import os
import numpy
 
# capture frames from a video
cap = cv2.VideoCapture('video1.MP4')
cap1 = cv2.VideoCapture('video.avi')
#cap2 =cv2.videoCapture("file name");
 
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
car_cascade1 = cv2.CascadeClassifier('cars.xml')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()
    ret, frames1 = cap1.read()
     
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frames1, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('video3',gray1)
    cv2.imshow('video4',gray)
     
 
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    cars1 = car_cascade1.detectMultiScale(gray1, 1.1, 1)
     
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
 
   # Display frames in a window 
        cv2.imshow('video1', frames)
    for (x,y,w,h) in cars1:
        cv2.rectangle(frames1,(x,y),(x+w,y+h),(0,0,255),2)
    
   # Display frames in a window 
        cv2.imshow('video2', frames1)
    
    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
     
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()
