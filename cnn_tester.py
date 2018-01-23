import cnn_trainer as cnn
import dataset_builder as db
import skin_reco
import numpy as np
import cv2

from keras.preprocessing.image import array_to_img, img_to_array

def main():
    # load neural network
    model = cnn.read_model('cache', 'architecture.json', 'weights.h5')

    # load face reco haar
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

    # init camera
    cap = cv2.VideoCapture(0)

    while(True):
        # get image from camera
        ret, frame = cap.read()
        cv2.rectangle(frame, (0,0), (300,300), (0,0,255), 1)
        area = frame[0:300, 0:300]

        # extract hand using skin color
        lower_range, upper_range = skin_reco.hsv_color_range_from_image(frame, face_cascade)
        if lower_range is not None and upper_range is not None:
            result = skin_reco.filter_skin(area, lower_range, upper_range)

            # suit the image for the network: reshape, normalize
            image = cv2.resize(result, (db.width, db.height))
            image = img_to_array(image)
            image = np.array(image, dtype="float") / 255.0
            image = image.reshape(1, db.width, db.height, db.channel)

            # use the model to predict the output
            output = model.predict(image)
            print(output)

            cv2.imshow('hand', result)

        # display
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
