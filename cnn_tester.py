import cnn_trainer as cnn
import dataset_builder as db
import numpy as np
import cv2

from keras.preprocessing.image import array_to_img, img_to_array

def main():
    # load neural network
    model = cnn.read_model('cache', 'architecture.json', 'weights.h5')

    # init camera
    cap = cv2.VideoCapture(0)

    while(True):
        # get image from camera
        ret, frame = cap.read()
        cv2.rectangle(frame, (0,0), (300,300), (0,0,255), 1)
        area = frame[0:300, 0:300]

        # apply changes on image like the dataset
        skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, db.lower_range, db.upper_range)
        mask = cv2.erode(mask, skinkernel, iterations = 1)
        mask = cv2.dilate(mask, skinkernel, iterations = 1)
        mask = cv2.GaussianBlur(mask, (15,15), 1)
        area = cv2.bitwise_and(hsv, hsv, mask = mask)
        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # suit the image for the network: reshape, normalize
        image = cv2.resize(area, (db.width, db.height))
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = image.reshape(1, db.width, db.height, db.channel)

        # use the model to predict the output
        output = model.predict(image)
        print(output)

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
