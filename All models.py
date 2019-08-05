"""

Cameron Fabbri

Simple neural network implementation for classifying images.
Simply provide the folder for which your images are stored in.

Folder structure should have images for each class in a seperate
folder. Example

images/
   cat/
      image1.jpg
      image2.jpg
      ...
   dog/
      ...

"""

import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# import time
import _pickle as pickle  # from python2 to 3
from random import sample
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from sklearn import preprocessing

print("Libs good")

SHAPE = (30, 30)

def extract_feature(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img

def read_files(directory, sample_sizes):
    print("Reading files...")
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            images = sample(images, sample_sizes[num_classes - 1])  # sample
            for image in images:
                s += 1
                label_list.append(d)
                feature_list.append(extract_feature(root + d + "/" + image))

    print(str(num_classes) + " classes")
    return np.asarray(feature_list), np.asarray(label_list), num_classes

def convertLabels(label_list):
    """
    Converts text labels to numbers, i.e cat -> 0, dog -> 1, ...
    """
    num_labels = len(label_list)

    pre = preprocessing.LabelEncoder()

    label_list = pre.fit_transform(label_list)

    return label_list


if __name__ == "__main__":

    model_name = 'nn'
    # model_name = 'svm'

    image_folder = "F:/Acad/research/fafar/RSO/nd_code/alderley/images/"
    if len(sys.argv) > 2:
        image_folder = sys.argv[1]
    #    print("Usage: python neural.py [image folder]")
    #    exit()

    # FRAMESA 16960, FRAMESB 14607
    sample_sizes = [10, 5]
    # generating two numpy arrays for features and labels
    features, labels, num_classes = read_files(image_folder, sample_sizes)

    if model_name == 'nn':
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.2, random_state=12)

        net = buildNetwork(SHAPE[0] * SHAPE[1] * 3, 15000,
                           num_classes, bias=True, outclass=SoftmaxLayer)

        train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
        test_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)

        for feature, label in zip(X_train, y_train):
            train_ds.addSample(feature, label)

        for feature, label in zip(X_test, y_test):
            test_ds.addSample(feature, label)

        # checking for model
        if os.path.isfile(model_name+ ".pkl"):
            print("Using previous model...")
            trainer = pickle.load(open(model_name+ ".pkl", "rb"))
        else:
            print("Training")
            trainer = BackpropTrainer(net, train_ds, momentum=0.1,
                                      verbose=True, weightdecay=0.01)
            trainer.train()

            print("Saving model...")
            pickle.dump(trainer, open(model_name+ ".pkl", "wb"))
    elif model_name == 'svm':
        # Splitting the data into test and training splits
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.2, random_state=42)
        # checking for model
        if os.path.isfile(model_name+ ".pkl"):
            print("Using previous model...")
            svm = pickle.load(open(model_name+ ".pkl", "rb"))
        else:
            print("Fitting")
            # Fitting model
            svm = SVC()
            svm.fit(X_train, y_train)

            print("Saving model...")
            pickle.dump(svm, open(model_name+ ".pkl", "wb"))

    # Test
    print("Testing...\n")
    correct_count = 0
    total_count = 0
    for feature, label in zip(X_test, y_test):
        if model_name == 'nn':
            prediction = net.activate(feature).argmax(axis=0)
        elif model_name == 'svm':
            feature = feature.reshape(1, -1)
            prediction = svm.predict(feature)[0]

        if prediction == label:
            correct_count += 1
        total_count += 1
    acc = (float(correct_count) / float(total_count)) * 100

    print('acc is {}'.format(acc))