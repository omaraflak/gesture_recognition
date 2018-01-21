from cnn_trainer import read_model, img_cols, img_rows, img_channels
from keras.preprocessing.image import array_to_img, img_to_array
import numpy as np
import cv2

# HSV skin color range (to be scaled...)
lower_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])

def main():
    # load neural network
    model = read_model('cache', 'architecture.json', 'weights.h5')

    # init camera
    cap = cv2.VideoCapture(0)

    while(True):
        # get image from camera
        ret, frame = cap.read()
        cv2.rectangle(frame, (0,0), (300,300), (0,0,255), 1)
        area = frame[0:300, 0:300]

        # get region of interest, convert to hsv etc.
        skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.erode(mask, skinkernel, iterations = 1)
        mask = cv2.dilate(mask, skinkernel, iterations = 1)
        mask = cv2.GaussianBlur(mask, (15,15), 1)
        area = cv2.bitwise_and(hsv, hsv, mask = mask)
        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # reshape image
        image = cv2.resize(area, (img_rows, img_cols))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = image.reshape(1, img_rows, img_cols, img_channels)

        # prediction
        res = model.predict(image)
        print(res)

        # display
        cv2.imshow('frame', frame)
        cv2.imshow('area', area)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
