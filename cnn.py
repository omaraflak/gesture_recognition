# imports
import os
import os.path as path
import numpy as np
import json
import cv2

import keras
from keras.preprocessing.image import img_to_array
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# constants
trainPath = "./gestures/train"
testPath = "./gestures/test"

img_channels = 1
img_rows, img_cols = 200, 200
nb_classes = 4
batch_sz = 32
nb_epoch = 5


def listdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

def get_img(path):
	image = cv2.imread(path, 0)
	image = img_to_array(image)
	return image

def get_output(i):
	out = []
	for j in range(nb_classes):
		if j==i:
			out.append(1)
		else:
			out.append(0)
	return out

def load_data(path):
	X_data = []
	Y_data = []

	classesFolders = listdir(path)
	nb = len(classesFolders)

	if nb!=nb_classes:
		print 'Number of classes doesnt match'
		exit()

	for i in range(nb_classes):
		files = listdir(os.path.join(path, classesFolders[i]))
		for fl in files:
			X_data.append(get_img(os.path.join(path, classesFolders[i], fl)))
			Y_data.append(get_output(i))

	X_data = np.array(X_data, dtype="float") / 255.0
	Y_data = np.array(Y_data)

	return X_data, Y_data


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[img_rows, img_cols, img_channels]))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_sz, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))


def save_model(model):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)


def read_model():
	model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
	model.load_weights(os.path.join('cache', 'model_weights.h5'))
	return model


def main():
	K.set_image_dim_ordering('th')


	x_train, y_train = load_data(trainPath)
	x_test, y_test = load_data(testPath)

	model = build_model()
	train(model, x_train, y_train, x_test, y_test)

	score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
	print('Score: ', score)

	save_model(model)
	# model = read_model()

if __name__ == '__main__':
    main()