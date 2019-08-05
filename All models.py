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
import random
from random import sample
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, TanhLayer, \
    SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
from sklearn import preprocessing

random.seed(30)
np.random.seed(110)


print("Libs good")
SHAPE = (30, 30)

def extract_feature(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img

def read_files(directory, sample_sizes, model_name):
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
    if model_name=='nn':
        label_list = convertLabels(label_list)
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
    random_state = 12

    image_folder = "F:/Acad/research/fafar/RSO/nd_code/alderley/images/"
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]

    # FRAMESA 16960, FRAMESB 14607
    sample_sizes = [100, 200]
    SHAPE = (30, 30)

    # HYP
    hidden_dim = 100
    bias_ = True
    # SoftmaxLayer, TanhLayer, SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
    hiddenclass_ = TanhLayer
    outclass_ = SoftmaxLayer
    num_epoch = 3
    learningrate_ = 0.01
    lrdecay_ = 1.0
    momentum_ = 0.1
    batchlearning_ = False
    weightdecay_ = 0.01
    # HYP

    # generating two numpy arrays for features and labels
    features, labels, num_classes = read_files(image_folder, sample_sizes, model_name)
    # Splitting the data into test and training splits
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2, random_state=random_state)

    f = open("result_hd_" + str(hidden_dim) +
             "b_"+str(bias_) + ".txt", "a")
    f.write(model_name+"\n")

    if model_name == 'nn':
        net = buildNetwork(SHAPE[0] * SHAPE[1] * 3, hidden_dim,
                           num_classes, bias=bias_, hiddenclass=hiddenclass_, outclass=outclass_)

        train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
        test_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)

        for feature, label in zip(X_train, y_train):
            train_ds.addSample(feature, label)

        for feature, label in zip(X_test, y_test):
            test_ds.addSample(feature, label)

        # checking for model
        if os.path.isfile(model_name+ ".pkl"):
            tmp = "Using previous "+model_name+ " model...\n"
            print(tmp)
            f.write(tmp)
            trainer = pickle.load(open(model_name+ ".pkl", "rb"))
        else:
            tmp = "Training " + model_name+"\n"
            print(tmp)
            f.write(tmp)
            trainer = BackpropTrainer(net, train_ds, learningrate=learningrate_, lrdecay=lrdecay_,
                                      momentum=momentum_, verbose=True, batchlearning=batchlearning_,
                                      weightdecay=weightdecay_)

            # trainer.train()
            trainer.trainEpochs(epochs=num_epoch)
            # trainer.trainOnDataset(dataset)
            # trainer.trainUntilConvergence(dataset=None, maxEpochs=None,
            #                               verbose=None, continueEpochs=10, validationProportion=0.25)

            # print("Saving model")
            # pickle.dump(trainer, open(model_name+ ".pkl", "wb"))
    elif model_name == 'svm':
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
    tmp = "Testing...\n"
    print(tmp)
    f.write(tmp)
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
    tmp = 'acc is {} \n'.format(acc)
    print(tmp)
    f.write(tmp)
    f.close()
