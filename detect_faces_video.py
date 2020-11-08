#USAGE
#python detect_faces_video1.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

import numpy as np 
import cv2
import argparse
from imutils.video import VideoStream
import imutils
import time

#constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,
    help="path to the Caffe pretrained model")
ap.add_argument("-c","--confidence", type=float , default= 0.5,
    help="minimun probability to filter weak detections")
args = vars(ap.parse_args())

#loading our serialized model from disk
print ("loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#intialize the video stream
print("loading video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0) #delay to start the stream

#looping over the frames in the video stream
while True:
    #grab frame from the video stream and resize to width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)

    #grabbing the frame dimensions and convert it  to a blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
        (300,300),(104.0,177.0,123.0))
    
    #passing the blob through the network and obtain predictions
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        #extracts the confidence associated with the prediction

        #filtering weak detections as per input confidence
        if confidence < args["confidence"]:
            continue

        #computing the co-ordinates of the bounding box for the detected object
        box = detections[0,0,i,3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
            
        text = "{:.2f}%".format(confidence*100)

        #drawing bounding box with the associated probability
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame,(startX, startY), (endX, endY),
            (0,255,0),2)
        cv2.putText(frame,text, (startX,y),
            cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 0, 255),2)
        
    #show the output frame
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) 

    #key 'q' to break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows
vs.stop()