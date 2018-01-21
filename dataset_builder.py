import numpy as np
import time
import cv2
import os

# path to save captures
data_path = 'gestures/train/class1'
file_format = 'png'

# HSV skin color range (to be scaled...)
lower_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])

# size of image to save
width, height, channel = 32, 32, 1
grayscale = True

def main():
    # create path if not exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(data_path, " has been created.")

    print("##############################")
    print("Press 'r' to start/stop record")
    print("Press 'q' to quit")
    print("##############################")

    # open camera
    cap = cv2.VideoCapture(0)
    capture = False

    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()

        # draw rectangle for area to capture
        cv2.rectangle(img, (0,0), (300,300), (0,0,255), 1)
        area = img[0:300, 0:300]

        # extract hand using skin color
        skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.erode(mask, skinkernel, iterations = 1)
        mask = cv2.dilate(mask, skinkernel, iterations = 1)
        mask = cv2.GaussianBlur(mask, (15,15), 1)
        result = cv2.bitwise_and(hsv, hsv, mask = mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # display results
        cv2.imshow('camera', img)
        cv2.imshow('hand', result)

        # start/stop capture
        if capture:
            filename = str(int(round(time.time() * 1000)))+"."+file_format
            pic = cv2.resize(result, (width, height))
            cv2.imwrite(os.path.join(data_path, filename), pic)

        # handle keyboard events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            capture = not capture
            print("start capture" if capture else "stop capture")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
