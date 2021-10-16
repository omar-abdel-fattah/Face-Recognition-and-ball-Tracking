import cv2
import numpy as np

backSub_MOG = cv2.createBackgroundSubtractorMOG2()
backSub2_KNN = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    fgMask = backSub2_KNN.apply(frame)
    kernel_erosion = np.ones((5, 5), np.uint8)  # Window size must be odd
    kernel_dilation = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(fgMask, kernel_erosion)
    dilation = cv2.dilate(erosion, kernel_dilation)
    M = cv2.moments(dilation)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])

        cY = int(M["m01"] / M["m00"])

        # put text and highlight the center
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(frame, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow('Eroded', erosion)
    cv2.imshow('dilated', dilation)

    keyboard = cv2.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
capture.release()
cv2.destroyAllWindows()