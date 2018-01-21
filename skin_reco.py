import numpy as np
import cv2

# load haar pretrained file
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

# camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # camera capture to grayscale for head detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # preparing kmeans clustering with k=2 i.e. searching 2 most dominant colors
        K = 2
        h,w,c = hsv.shape
        Z = hsv.reshape((h*w, c))
        Z = np.float32(Z)

        # apply kmeans on face
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # print most dominant HSV color
        print(center[0].astype(int))

    # display camera preview
    cv2.imshow('camera', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
