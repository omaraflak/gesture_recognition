import os
import os.path as path
import numpy as np
import json
import cv2

import dataset_builder as db

import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# path of training data and testing data
trainPath = 'gestures/train'
testPath = 'gestures/test'

# number of classes/categories of network output (e.g. car, chicken, human --> 3)
nb_classes = 3

# how many images to process before applying gradient correction
batch_sz = 32

# how many times the network should train on the whole dataset
nb_epoch = 40

# how many images to generate per image in datasets
nb_gen = 20

# list files and folders in a given directory
def listdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

# loads an opencv image from a filepath
def get_img(path):
    image = cv2.imread(path, 0) if db.grayscale else cv2.imread(path, db.channel)
    image = cv2.resize(image, (db.width, db.height))
    image = img_to_array(image)
    image = image.reshape(db.width, db.height, db.channel)
    return image

# get output vector to train network
def get_output(i):
	out = []
	for j in range(nb_classes):
		if j==i:
			out.append(1)
		else:
			out.append(0)
	return out

# shuffle two arrays in the same way
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# use keras to generate more data from existing images
def generate_data(path):
	datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

	classesFolders = listdir(path)
	for folder in classesFolders:
		files = listdir(os.path.join(path, folder))
		for fl in files:
			img = get_img(os.path.join(path, folder, fl))
			img = img.reshape(1, db.width, db.height, db.channel)
			i = 0
			for batch in datagen.flow(img, batch_size=1, save_to_dir=os.path.join(path, folder), save_prefix='genfile', save_format=db.file_format):
				i += 1
				if i > nb_gen:
					break

# load data from dataset folder
# the dataset folder should include a different folder for each class e.g.
# --- dataset/
#       --- cars/
#           - car1.png
#           - car2.png
#       --- humans/
#           - human1.png
#           - human2.png
#       --- chickens/
#           - chicken1.png
#           - chicken2.png
def load_data(path):
	X_data = []
	Y_data = []
	mapping = []

	classesFolders = listdir(path)
	nb = len(classesFolders)

	if nb!=nb_classes:
		print('Number of classes doesnt match')
		exit()

	for i in range(nb_classes):
		files = listdir(os.path.join(path, classesFolders[i]))
		mapping.append(classesFolders[i]+' : '+str(i))
		for fl in files:
			X_data.append(get_img(os.path.join(path, classesFolders[i], fl)))
			Y_data.append(get_output(i))

	X_data = np.array(X_data, dtype="float") / 255.0
	Y_data = np.array(Y_data)

	X_data, Y_data = shuffle_in_unison(X_data, Y_data)

	return X_data, Y_data, mapping

# build convolutional neural network
def build_model():
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[db.width, db.height, db.channel]))
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
	model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Flatten())
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))

	return model

# train model with data
def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_sz, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

# save network model and network weights into files
def save_model(model, network_path, network_model, network_weights):
    if not os.path.isdir(network_path):
        os.mkdir(network_path)
    open(os.path.join(network_path, network_model), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, network_weights), overwrite=True)

# load network model and network weights from files
def read_model(network_path, network_model, network_weights):
	model = model_from_json(open(os.path.join(network_path, network_model)).read())
	model.load_weights(os.path.join(network_path, network_weights))
	return model

def main():
    # generate data
    generate_data(trainPath)

    # Load data
    x_train, y_train, mapping = load_data(trainPath)
    x_test, y_test, mapping2 = load_data(testPath)

    # Create network, train it, save it
    model = build_model()
    train(model, x_train, y_train, x_test, y_test)
    save_model(model, 'cache', 'architecture.json', 'weights.h5')

    # Evaluate model on test data
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
    main()
