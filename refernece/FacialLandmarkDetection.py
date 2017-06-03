import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy
import matplotlib.pyplot as plot

predictor_path = "./shape_predictor_68_face_landmarks.dat"  #Face model
detector = dlib.get_frontal_face_detector()  #Get the face detector
predictor = dlib.shape_predictor(predictor_path)  #Get the face predictor

cap = cv2.VideoCapture(0)  #capture the camera
win = dlib.image_window()
counter =0
while(1):
    # Too slow, skip some frames
    counter = counter + 1
    print counter
    if counter < 3000:
        continue
    counter = 0
    print counter
    ret, img = cap.read()  #Get each frame
    print("Processing ......")

    win.imshow(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
        win.add_overlay(dets)
        os.system("pause")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
