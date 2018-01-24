import skin_reco
import numpy as np
import time
import cv2
import os

# path to save captures
dataset_folder = 'gestures'
class_name = 'class3'
file_format = 'png'

# training and testing folder
train_folder = 'train'
test_folder = 'test'

# size of image to save
width, height, channel = 32, 32, 1
grayscale = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, " has been created.")

def timestamped_filename(file_format):
    return str(int(round(time.time() * 1000)))+"."+file_format

def main():
    trainPath = os.path.join(dataset_folder, train_folder, class_name)
    testPath = os.path.join(dataset_folder, test_folder, class_name)

    # create paths if not exists
    check_path(trainPath)
    check_path(testPath)

    # load face reco haar
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

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
        cv2.rectangle(img, (100,100), (300,300), (0,0,255), 1)
        area = img[100:300, 100:300]

        # extract hand using skin color
        lower_range, upper_range = skin_reco.hsv_color_range_from_image(img, face_cascade)
        if lower_range is not None and upper_range is not None:
            result = skin_reco.filter_skin(area, lower_range, upper_range)
            cv2.imshow('result', result)

        # display results
        cv2.imshow('camera', img)

        # start/stop capture
        if capture:
            filename = timestamped_filename(file_format)
            pic = cv2.resize(result, (width, height))
            cv2.imwrite(os.path.join(trainPath, filename), pic)

        # handle keyboard events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            capture = not capture
            if capture:
                print("start capture")
                filename = timestamped_filename(file_format)
                pic = cv2.resize(result, (width, height))
                cv2.imwrite(os.path.join(testPath, filename), pic)
            else:
                print("stop capture")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
