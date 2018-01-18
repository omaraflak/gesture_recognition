import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # draw rectangle
    cv2.rectangle(frame, (0,0), (300,300), (0,0,255), 1)

    # get roi
    area = frame[0:300, 0:300]

    # roi to gray
    area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('area', area)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
