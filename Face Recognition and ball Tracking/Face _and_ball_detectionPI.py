# sys.path.append('/usr/local/lib/python3.7/dist-packages/cv2')
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import statistics
import numpy as np
import time
import serial
import face_recognition
import sys
import pickle
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(31, GPIO.OUT)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.LOW)

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# haarcascade
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

detector = cv2.CascadeClassifier(cascade)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

greenLower = (135, 135, 135)
greenUpper = (255, 255, 255)
Frame_counter = 0;
limit = 0
L = []
limitx = []
boxes = []
Xboxes = []
akey = 0
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    global akey
    global limit
    global limitx
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                akey = 0
                objectInfo.append([box, className])

                boxes.append(box[1])

                if (draw):

                    for a in range(len(objectInfo)):
                        L.append(((((objectInfo[a][0][0] - objectInfo[a][0][1])2) + (
                                    (objectInfo[a][0][2] - objectInfo[a][0][3]) ** 2))0.5))


                print(box[0])

                if (abs(box[0] - cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2) <= 110:
                    print('Approach')
                    GPIO.output(31, GPIO.HIGH)
                    time.sleep(0.5)
            else:
                GPIO.output(31, GPIO.LOW)
                akey += 1
                if (akey % 20) == 0:
                    akey = 0
                    boxes.clear()
                    Xboxes.clear()
        L.clear()

    return img, objectInfo


cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
print(cap.get(cv2.CAP_PROP_FPS))

while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    success, img = cap.read()
    frame = cap.read()
    frame = imutils.resize(frame, width=500)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Frame_counter = Frame_counter + 1;
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # if face is not recognized, then print Unknown
        GPIO.output(7, GPIO.LOW)
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # setting GPIO PIN 7 high
            GPIO.output(7, GPIO.HIGH)
            time.sleep(0.5)
            # setting GPIO PIN 7 low
            GPIO.output(7, GPIO.LOW)

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            # If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)
                # line = ser.readline().decode('utf-8').rstrip()
                # print(line)
            # time.sleep(1)
        # update the list of names

        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)

    # display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

    # stop the timer and display FPS information
    ### converting to bounding boxes from polygon
    if (Frame_counter % 20 == 0):
        result, objectInfo = getObjects(img, 0.3, 0.5, objects=['sports ball'])
    cv2.namedWindow('Video Life2Coding', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Video Life2Coding', img)

    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(-1)
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))