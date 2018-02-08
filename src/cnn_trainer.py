import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

import dataset_builder as db

import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from sklearn.model_selection import train_test_split

# how many images to process before applying gradient correction
batch_sz = 32

# how many times the network should train on the whole dataset
nb_epoch = 200

# how many images to generate per image in datasets
nb_gen = 20

# create path if not exists
def create_ifnex(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# exit program is path if not exists
def exit_ifnex(directory):
    if not os.path.exists(directory):
        print(directory, 'does not exist')
        exit()

# loads an opencv image from a filepath
def get_img(path):
    image = cv2.imread(path, 0) if db.grayscale else cv2.imread(path, db.channel)
    image = cv2.resize(image, (db.width, db.height))
    image = img_to_array(image)
    image = image.reshape(db.width, db.height, db.channel)
    return image

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

	classesFolders = os.listdir(path)
	for folder in classesFolders:
		files = os.listdir(os.path.join(path, folder))
		for fl in files:
			img = get_img(os.path.join(path, folder, fl))
			img = img.reshape(1, db.width, db.height, db.channel)
			i = 0
			for batch in datagen.flow(img, batch_size=1, save_to_dir=os.path.join(path, folder), save_prefix='genfile', save_format=db.file_format):
				i += 1
				if i > nb_gen:
					break

# load data from dataset folder
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
def load_data(dataset_path):
    x_data = []
    y_data = []
    labels = []

    classes = os.listdir(dataset_path)
    for i in range(len(classes)):
        files = os.listdir(os.path.join(dataset_path, classes[i]))
        labels.append(classes[i])
        for fl in files:
            x_data.append(get_img(os.path.join(dataset_path, classes[i], fl)))
            y_data.append(i)

    x_data = np.array(x_data, dtype="float") / 255.0
    y_data = np.array(y_data)

    y_data = keras.utils.np_utils.to_categorical(y_data)
    return x_data, y_data, labels

# split dataset into training and testing
def split_dataset(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
    return x_train, y_train, x_test, y_test

# build convolutional neural network
def build_model(nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[db.height, db.width, db.channel]))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

# train model with data
def train(model, x_train, y_train):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_sz, epochs=nb_epoch, verbose=1, validation_split=0.3)
    return history

# save network model and network weights into files
def save_model(model, network_path):
    create_ifnex(network_path)
    open(os.path.join(network_path, 'architecture.json'), 'w').write(model.to_json())
    model.save_weights(os.path.join(network_path, 'weights.h5'), overwrite=True)

# load network model and network weights from files
def read_model(network_path):
    exit_ifnex(network_path)
    model = model_from_json(open(os.path.join(network_path, 'architecture.json')).read())
    model.load_weights(os.path.join(network_path, 'weights.h5'))
    return model

# export model for mobile devices (tensorflow lite)
def export_model_for_mobile(dst, model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, dst, \
        model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), dst + '/' + model_name + '.chkp')

    freeze_graph.freeze_graph(dst + '/' + model_name + '_graph.pbtxt', None, \
        False, dst + '/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        dst + '/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(dst + '/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(dst + '/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

def plot_history(history):
    #  Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def main():
    # generate data
    generate_data(db.dataset_folder)

    # Load data, split data
    x_data, y_data, labels = load_data(db.dataset_folder)
    x_train, y_train, x_test, y_test = split_dataset(x_data, y_data)

    # Create network, train it, save it
    nb_classes = len(os.listdir(db.dataset_folder))
    model = build_model(nb_classes)
    history = train(model, x_train, y_train)
    model.summary()
    save_model(model, '../model')

    # Export model for tensorflow lite + write labels
    export_model_for_mobile('../out', 'convnet', "conv2d_1_input", "dense_2/Softmax")
    fl = open('../out/labels.txt', 'w')
    for item in labels:
        fl.write("%s\n" % item)

    # Evaluate model on test data
    scores = model.evaluate(x_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # display graphs
    plot_history(history)

if __name__ == '__main__':
    main()
