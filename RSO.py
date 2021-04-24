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

MJA:
used for Day Night our algorithm
"""

import sys
import time
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# import time
import _pickle as pickle  # from python2 to 3
import random
from random import sample
## PY
# from pybrain.datasets import SupervisedDataSet
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.structure.modules import SoftmaxLayer, TanhLayer, \
#     SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
from sklearn import preprocessing
import pandas as pd
from shutil import copyfile
##
import csv

# import tensorflow as tf

# import torch.nn as nn
# import torch.nn.functional as F

import ConvNN_t
from base_mnist import mymnist
from utils import write_csv

random.seed(30)
np.random.seed(110)
# print("Libs good")
SHAPE = (30, 30)
# model_name = 'nn'
# model_name = 'svm'

# import scipy
# scipy.stats.skew(X_train)


def extract_feature(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img


def sample_folder(images_dir, sample_sizes):
    # from shutil import copyfile
    print("Build a sample folder")
    # root2 = images_dir +"_"+ str(sample_sizes[0])+","+str(sample_sizes[1])+"/"
    root2 = images_dir + "_" + str(sample_sizes)+ "/"
    make_dir(root2)
    s = 0
    num_classes = 0
    for root, dirs, files in os.walk(images_dir+"/"):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            if sample_sizes[0] >0:
                images = sample(images, sample_sizes[num_classes - 1])  # sample
            for image in images:
                s += 1
                make_dir(root2 + d + "/")
                copyfile(root + d + "/" + image, root2 + d + "/" + image)
    print("made {} folders and copied {} files".format(num_classes, s))
    return


def make_dir(dir):
    if not os.path.exists(dir):
        print("dir ( {} ) is made ".format(dir))
        os.mkdir(dir)


def test_train_sep(images_dir, test_precs):

    print("test_train_sep folder {}".format(images_dir))
    # root2 = images_dir +"_"+str(test_precs[0])+","+str(test_precs[1])+"/"
    root2 = images_dir + "_" + str(test_precs) + "/"
    make_dir(root2)
    s = 0
    num_classes = 0
    for root, dirs, files in os.walk(images_dir+"/"):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)

            ln_ = len(images)
            random.shuffle(images)
            ln_ = int(ln_*test_precs[num_classes - 1])
            images_tes = images[:ln_]
            images_tr = images[ln_:]

            source_dir = root + d + "/"

            tr_dir = root2 + "tr/"
            make_dir(tr_dir)
            for image in images_tr:
                s += 1
                tr_dir_im = root2 + "tr/" + d + "/"
                make_dir(tr_dir_im)
                copyfile(source_dir+ image, tr_dir_im + image)

            tes_dir = root2 + "tes/"
            make_dir(tes_dir)
            for image in images_tes:
                s += 1
                tes_dir_im = root2 + "tes/" + d + "/"
                make_dir(tes_dir_im)
                copyfile(source_dir + image, tes_dir_im + image)
    print("made {} folders and copied {} files".format(num_classes*2, s))
    return tr_dir, tes_dir


# def read_files(directory, sample_sizes, model_name):
def read_files(directory, model_name):
    print("Reading files in {}".format(directory))
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0

    for class_fldr in os.listdir(directory):
        if class_fldr not in classes_labels:
            continue
        num_classes += 1
        images = os.listdir(directory + class_fldr)

        source_dir = directory + class_fldr + "/"
        for image in images:
            s += 1
            label_list.append(class_fldr)
            feature_list.append(extract_feature(source_dir + image))
    # for root, dirs, files in os.walk(directory):
    #     for d in dirs:
    #         if d == "divided":
    #             continue
    #         num_classes += 1
    #         images = os.listdir(root + d)
    #         # if sample_sizes[0] > 0:
    #         #     images = sample(images, sample_sizes[num_classes - 1])  # sample
    #         # print(root + d)
    #         for image in images:
    #             s += 1
    #             label_list.append(d)
    #             feature_list.append(extract_feature(root + d + "/" + image))
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


def divide_with_prec(points_list_file, dire, size_of_trs):

    # import pandas as pd
    # from shutil import copyfile
    df = pd.read_csv(points_list_file)
    # num_tr_folders = len(df['prec day'])
    s = 0
    divide_dir = dire + "divided/"
    make_dir(divide_dir)
    for cntr, day_prec in enumerate(df['day prec']):
        num_classes = 0
        for class_fldr in os.listdir(dire):
            if (class_fldr == "FRAMESB"):
                prec = day_prec
            elif (class_fldr == "FRAMESA"):
                prec = 1 - day_prec
            else:
                continue

            num_classes += 1
            images = os.listdir(dire + class_fldr)

            ln_ = len(images)
            random.shuffle(images)
            ln_ = min(int(prec * size_of_trs), ln_)
            print("{} images picked".format(ln_))
            images_ = images[:ln_]

            source_dir = dire + class_fldr + "/"

            division_dir = divide_dir + str(cntr) + "/"
            make_dir(division_dir)
            division_dir_d = divide_dir + str(cntr) + "/" + class_fldr + "/"
            make_dir(division_dir_d)

            for image in images_:
                s += 1
                copyfile(source_dir + image, division_dir_d + image)

    print("copied {} files".format(s))
    return


def eval(divide_files_dir, division_num, test_precs, model_name, X, Y, net, svm, f, tr_):
    # tr acc
    # tmp = "Eval on {}\n".format(divide_files_dir)
    # print(tmp)
    # f.write(tmp)
    # make_dir(divide_files_dir + "res/")
    # if tr_:
    #     pred_file = open(divide_files_dir + "res/tr_" + str(division_num) + "_preds"
    #                      + str(test_precs) + "_" + model_name + ".csv", 'a')
    # else:
    #     pred_file = open(divide_files_dir + "res/tes_" + str(division_num) + "_preds"
    #                      + str(test_precs) + "_" + model_name + ".csv", 'a')

    # writer_pred_file = csv.writer(pred_file)

    correct_count = 0
    total_count = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for feature, label in zip(X, Y):
        if model_name == 'nn':
            prediction = net.activate(feature).argmax(axis=0)
        elif model_name == 'svm':
            feature = feature.reshape(1, -1)
            prediction = svm.predict(feature)[0]

        if prediction == label:
            correct_count += 1
            if label==1:
                tp+=1
            else:
                tn+=1
        else:
            if label==1:
                fn+=1
            else:
                fp+=1

        # row = [total_count, label, prediction, correct_count]
        # writer_pred_file.writerow(row)
        total_count += 1

    # acc, prec, recall, f1 = 0,0,0,0
    acc = float(correct_count) / max(float(total_count), 1)
    prec = float(tp)/max(float(tp+fp), 1)
    recall = float(tp)/max(float(tp+fn), 1)
    f1 = float(2*tp)/max(float(2*tp+fp+fn), 1)

    # tmp = 'Acc, prec, recal, f1 on ' +str("tr " if tr_ else "tes ")+\
    #       str(division_num)+" are {}, {}, {}, {} \n".format(acc, prec, recall, f1)
    # print(tmp)
    # f.write(tmp)
    # pred_file.close()

    return acc, prec, recall, f1


# class nn_hold():
#     def __init__(self, bias_, hiddenclass_, outclass_, momentum_, batchlearning_ ):
#         self.bias_ = bias_
#         self.hiddenclass_ = hiddenclass_
#         self.outclass_ = outclass_
#         self.momentum_ = momentum_
#         self.batchlearning_ = batchlearning_
#
#     def nn_run(self, hidden_dim, num_epoch, learningrate_, lrdecay_, weightdecay_, num_classes, X_train, y_train):
#         # # NN HYP
#         # # hidden_dim = 100
#         # bias_ = True
#         # # SoftmaxLayer, TanhLayer, SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
#         # hiddenclass_ = TanhLayer
#         # outclass_ = SoftmaxLayer
#         # # num_epoch = 4
#         # # if len(sys.argv)>0:
#         # #     num_epoch = int(sys.argv[1])
#         # # learningrate_ = 0.01
#         # # lrdecay_ = 1.0
#         # momentum_ = 0.1
#         # batchlearning_ = False
#         # # weightdecay_ = 0.01
#         # # NN HYP
#
#         net = buildNetwork(SHAPE[0] * SHAPE[1] * 3, hidden_dim,
#                            num_classes, bias=self.bias_, hiddenclass=self.hiddenclass_, outclass=self.outclass_)
#
#         train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
#         test_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
#
#         if batch_size ==0:
#             # for feature, label in zip(X_train, y_train):
#             for feature, label in zip(X_train, y_train):
#                 train_ds.addSample(feature, label)
#
#             # for feature, label in zip(X_test, y_test):
#             #     test_ds.addSample(feature, label)
#
#             # checking for model
#             if os.path.isfile("models/" + model_name + ".pkl"):
#                 tmp = "Using previous " + model_name + " model...\n"
#                 print(tmp)
#                 f.write(tmp)
#                 trainer = pickle.load(open("models/" + model_name + ".pkl", "rb"))
#             else:
#                 # tmp = "Training " + model_name + " on set " + str(division_num) + "\n"
#                 # print(tmp)
#                 # f.write(tmp)
#                 trainer = BackpropTrainer(net, train_ds, learningrate=learningrate_, lrdecay=lrdecay_,
#                                           momentum=self.momentum_, verbose=True, batchlearning=self.batchlearning_,
#                                           weightdecay=weightdecay_)
#                 # different trainig calls
#                 # trainer.train()
#                 trainer.trainEpochs(epochs=num_epoch)
#                 # trainer.trainOnDataset(dataset)
#                 # trainer.trainUntilConvergence(dataset=None, maxEpochs=None,
#                 #                               verbose=None, continueEpochs=10, validationProportion=0.25)
#                 # different trainig calls
#
#                 # print("Saving model")
#                 # pickle.dump(trainer, open("models/"+ model_name+ ".pkl", "wb"))
#         elif batch_size>0:
#             trainer = BackpropTrainer(net, learningrate=learningrate_, lrdecay=lrdecay_,
#                                       momentum=self.momentum_, verbose=True,
#                                       batchlearning=self.batchlearning_,
#                                       weightdecay=weightdecay_)
#             for epoch in range(num_epoch):
#                 print("\n epoch {}".format(epoch))
#                 for i in range(X_train.shape[0] // batch_size):
#                     X_ = X_train[i * batch_size:(i + 1) * batch_size][:]
#                     y_ = y_train[i * batch_size:(i + 1) * batch_size]
#
#                     tmp = "epoch {}, batch {}".format(epoch, i)
#                     print(tmp)
#                     f.write(tmp)
#
#                     train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
#
#                     for feature, label in zip(X_, y_):
#                         train_ds.addSample(feature, label)
#
#                     # train_ds.batches("batches", batch_size)
#
#                     # for feature, label in zip(X_test, y_test):
#                     #     test_ds.addSample(feature, label)
#
#                     # checking for model
#                     if os.path.isfile("models/" + model_name + ".pkl"):
#                         tmp = "Using previous " + model_name + " model...\n"
#                         print(tmp)
#                         f.write(tmp)
#                         trainer = pickle.load(open("models/" + model_name + ".pkl", "rb"))
#                     else:
#                         # tmp = "Training " + model_name + " on set " + str(division_num) + "\n"
#                         # print(tmp)
#                         # f.write(tmp)
#                         # trainer = BackpropTrainer(net, learningrate=learningrate_, lrdecay=lrdecay_,
#                         #                           momentum=self.momentum_, verbose=True,
#                         #                           batchlearning=self.batchlearning_,
#                         #                           weightdecay=weightdecay_)
#                         # different trainig calls
#                         # trainer.train()
#                         trainer.trainOnDataset(train_ds)
#                         # trainer.trainOnDataset(dataset)
#                         # trainer.trainUntilConvergence(dataset=None, maxEpochs=None,
#                         #                               verbose=None, continueEpochs=10, validationProportion=0.25)
#                         # different trainig calls
#
#                         # print("Saving model")
#                         # pickle.dump(trainer, open("models/"+ model_name+ ".pkl", "wb"))
#
#                     # tmp = eval(" ", " ", test_precs, model_name,
#                     #            X_train, y_train, net, svm, f, tr_=True)
#                     # print("eval {}".format(tmp))
#
#         return net


def svm_run(X_train, y_train):

    # Hyps
    C_ = 1.0  # Penalty parameter C of the error term.
    kernel_ = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree_ = 3  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    gamma_ = 'scale'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Set gamma explicitly to 'auto' or 'scale' to avoid this warning
    coef0_ = 0.0  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    shrinking_ = True  # Whether to use the shrinking heuristic.
    probability_ = False  # Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
    tol_ = 0.001  # Tolerance for stopping criterion.
    max_iter_ = -1  # Hard limit on iterations within solver, or -1 for no limit.
    # Hyps

    f = open("res/result_krnl_" + str(kernel_) +
             "tol_" + str(tol_) + ".txt", "a")
    f.write(model_name + "\n")

    # checking for model
    if os.path.isfile("models/" + model_name + ".pkl"):
        print("Using previous model...")
        svm = pickle.load(open("models/" + model_name + ".pkl", "rb"))
    else:
        print("Fitting")
        # Fitting model
        svm = SVC()
        svm = SVC(C=C_, kernel=kernel_, degree=degree_, gamma=gamma_, coef0=coef0_,
                  shrinking=shrinking_, probability=probability_, tol=tol_,
                  verbose=False, max_iter=max_iter_)

        svm.fit(X_train, y_train)

        # print("Saving model...")
        # pickle.dump(svm, open("models/"+ model_name+ ".pkl", "wb"))

    return svm


def objective_(hyps):
    hidden_dim = hyps['hidden_dim']+ hidden_dim_l
    learningrate_ = hyps['learningrate_']
    lrdecay_ = hyps['lrdecay_']
    weightdecay_ = hyps['weightdecay_']

    net = nn_hold_.nn_run(hidden_dim, num_epoch, learningrate_,
                          lrdecay_, weightdecay_, num_classes, X_train, y_train)

    eval_ = eval(tr_dir, "hpopt", test_precs, model_name, X_train, y_train, net, svm, out_file, tr_=True)

    return eval_[0] # acc


def objective_cnn(hyps):

    CNN_w.batch_size = hyps['batch_size']+ batch_size_l
    CNN_w.lr = hyps['lr']
    CNN_w.krnl_2 = hyps['krnl_2'] + krnl_2_l
    CNN_w.mx_krnl_2 =  hyps['mx_krnl_2'] + mx_krnl_2_l

    CNN_w.train_reader()

    tr_acc, tes_acc = CNN_w.trainer()

    return tes_acc


if __name__ == "__main__":

    tr_tes_sep = 0
    sample_folder_build = 0
    divide_file = 0
    hyperopt_use = 0
    # hyperopt_use = 1
    hype_given = 0
    # hype_given = 1
    RSO_use = 0
    RSO_use = 1
    mnist_on = 1
    if len(sys.argv) > 1:
        if sys.argv[2]=="tr_tes_sep":
            tr_tes_sep = True
        else:
            tr_tes_sep = False
        if sys.argv[2]=="sample_folder_build":
            sample_folder_build = True
        else:
            sample_folder_build = False
        if sys.argv[2]=="divide_file":
            divide_file = True
        else:
            divide_file = False
        if sys.argv[2]=="hyperopt_use":
            hyperopt_use = True
        else:
            hyperopt_use = False

    images_dir = "images"
    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images_[500,550]"

    make_dir("res/")

    size_of_trs = 6000
    # size_of_trs = 50

    mnist_tr_size = 10000
    # mnist_tr_size = 100
    mnist_tes_size = 3000
    # mnist_tes_size = 30

    # init net and svm
    net = None
    svm = None
    model_name = 'nn'
    # model_name = 'svm'
    random_state = 12

    num_epoch = 5
    if len(sys.argv) > 1:
        num_epoch = int(sys.argv[1])

    batch_size = 0

    # # hyps:
    # # int:
    # # hidden_dim[20, 200]
    # hidden_dim_l = 20
    # hidden_dim_u = 200
    # #
    # # Real:
    # # learningrate_[1e-5, 0.1]
    # learningrate_l = 1e-5
    # learningrate_u = 0.1
    # # lrdecay_[1e-2, 1e-1]
    # lrdecay_l = 1e-2
    # lrdecay_u = 1e-1
    # # weightdecay_[1e-3, 0.9]
    # weightdecay_l = 1e-3
    # weightdecay_u = 0.9

    # points_list_file = "Design-Data.csv"
    # points_list_file = "Design-Data-small.csv"
    points_list_file = "LHS-data.csv"  # day night
    points_list_file = "lhs-mnist.csv"  # mnist
    # points_list_file = "lhs-mnist-small.csv"  # mnist
    test_precs_file = "Test-Data.csv"

    # FRAMESA (night) 16960, FRAMESB (day) 14607 # in CNN FRAMESB is 1
    classes_labels = ["FRAMESA", "FRAMESB"]
    # sample_sizes = [100, 200]
    # sample_sizes = [500, 550]
    sample_sizes = [-1, -1]  # -1 for not sampling
    test_precs= [.2, .2]
    SHAPE = (30, 30)

    if sample_folder_build:
        # Build a sample folder or separate test and train
        sample_folder(images_dir, sample_sizes)
        exit()

    # sep tr tes
    if tr_tes_sep:
        test_train_sep(images_dir, test_precs)
        exit()

    # generating two numpy arrays for features and labels
    # features, labels, num_classes = read_files(image_folder, sample_sizes, model_name)
    # Splitting the data into test and training splits
    # test_prec = 0
    # if test_prec > 0:
    #     X_train, X_test, y_train, y_test = train_test_split(features, labels,
    #                                                         test_size=0, random_state=random_state)
    # else:
    #     tr_st, tr_end, tes_st, tes_end = 0,100, 20,30
    #     X_train, y_train = features[tr_st:tr_end+1,:], labels[tr_st:tr_end+1]
    #     X_test, y_test = features[tes_st:tes_end+1], labels[tes_st:tes_end+1]

    if divide_file:
        dire = images_dir +"_"+ str(test_precs) + "/tr/"
        divide_with_prec(points_list_file, dire, size_of_trs)
        exit()

    # read_files("F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]_[0.2, 0.2]/tr/", model_name)
    # exit()

    tes_dir = images_dir + "_" + str(test_precs) + "/tes/"

    divide_files_dir = images_dir + "_" + str(test_precs) + "/divided/"
    division_num = 0

    # # NN HYP
    # hidden_dim = 100
    # bias_ = True
    # # SoftmaxLayer, TanhLayer, SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
    # hiddenclass_ = TanhLayer
    # outclass_ = SoftmaxLayer
    # # num_epoch = 4
    # # if len(sys.argv)>0:
    # #     num_epoch = int(sys.argv[1])
    # # learningrate_ = 0.01
    # # lrdecay_ = 1.0
    # momentum_ = 0.1
    # batchlearning_ = True
    # weightdecay_ = 0.01
    # # NN HYP

    # hypopt

    if hyperopt_use:
        from hyperopt import hp
        from hyperopt import fmin, tpe, space_eval
        max_eval_hpopt = 17
        max_eval_hpopt = 3
        if len(sys.argv) > 1:
            max_eval_hpopt = int(sys.argv[3])

        out_file = open("res/result_" + str(test_precs) + "_" + model_name + "_hyopt" + str(max_eval_hpopt) +
                 "_epo" + str(num_epoch) + ".txt", "a")
        out_file.write(model_name + "\n")

        f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
        writer_f_all = csv.writer(f_all)

        tr_dir = images_dir + "_" + str(test_precs) + "/tr/"

        print("hyperopt started ")

        test_dataset = None
        test_load = None
        st_time = time.time()

        # find best hyps
        if mnist_on:
            hyp_rngs = {'lr': (1e-4, 1e-1), 'batch_size':(10, 64), 'fc_size':(30, 200), 'mxp_krnl':(2, 10)}
            mymnistTmp = mymnist(hyp_rngs=hyp_rngs)
            mymnistTmp.load_dataset(tr_ss=mnist_tr_size, tes_ss=mnist_tes_size)
            objective_cnn = mymnistTmp.evaluate_model
            space_ = {'batch_size': hp.randint('batch_size', hyp_rngs['batch_size'][1] - hyp_rngs['batch_size'][0] + 1),
                      'lr': hp.uniform('lr', hyp_rngs['lr'][0], hyp_rngs['lr'][1]),
                      'fc_size': hp.randint('fc_size',  hyp_rngs['fc_size'][1] - hyp_rngs['fc_size'][0] + 1),
                      'mxp_krnl': hp.randint('mxp_krnl', hyp_rngs['mxp_krnl'][1] - hyp_rngs['mxp_krnl'][0] + 1)}
        else:
            im_size = 64
            batch_size = 1000; batch_size_l = 50
            batch_size_u = 400
            lr = 0.0001; lr_l = 1e-4; lr_u = 1e-2
            krnl_1 = 5; krnl_2 = 5; krnl_2_l = 2; krnl_2_u = 20
            mx_krnl_1 = 2; mx_krnl_2 = 2; mx_krnl_2_l = 2
            mx_krnl_2_u = 10; num_epochs = 1
            CNN_w = ConvNN_t.CNN_wrap(im_size, batch_size, lr, krnl_1, krnl_2, mx_krnl_1,
                                  mx_krnl_2, num_epochs, tr_dir + "/", tes_dir)
            if test_dataset is not None:
                CNN_w.test_dataset = test_dataset
                CNN_w.test_load = test_load
                print("used previous test read from {}".format(tes_dir))
            else:
                CNN_w.test_reader()
                test_dataset = CNN_w.test_dataset
                test_load = CNN_w.test_load
                print("test loaded form {}".format(tes_dir))
            # CNN_w.test_reader()
            print("train hypopt started on {}".format(tr_dir))
            space_ = {'batch_size': hp.randint('batch_size', batch_size_u - batch_size_l+1),
                      'lr': hp.uniform('lr', lr_l, lr_u),
                      'krnl_2': hp.randint('krnl_2', krnl_2_u-krnl_2_l+1),
                      'mx_krnl_2': hp.randint('mx_krnl_2', mx_krnl_2_u-mx_krnl_2_l+1)}
            # try the objective_cnn
            # hyps = {'batch_size':50, 'lr': 0.001, 'krnl_2':5, 'mx_krnl_2':4}
            # objective_cnn(hyps)

        # minimize the objective over the space
        best_hyp = fmin(objective_cnn, space_, algo=tpe.suggest, max_evals=max_eval_hpopt)
        tmp = "\n optimal hyps with tpe hypopt are {}\n".format(best_hyp)
        out_file.write(tmp)
        print(tmp)

        # get the best hype performance and stats
        if mnist_on:
            tes_acc = mymnistTmp.evaluate_model(best_hyp)
            tr_acc = mymnistTmp.tr_eval()
            tr_data_ave, tr_data_std = mymnistTmp.tr_ave, mymnistTmp.tr_std
            tes_data_ave, tes_data_std = mymnistTmp.tes_ave, mymnistTmp.tes_std
            hyp_opt_time = time.time() - st_time
            row = ["Hypeopt", mymnistTmp.img_size, best_hyp['batch_size'], best_hyp['lr'], best_hyp['fc_size'],
                   best_hyp['mxp_krnl'],
                   tr_data_ave, tr_data_std, tr_acc, "",
                   tes_data_ave, tes_data_std, tes_acc,
                   hyp_opt_time]
        else:
            CNN_w.batch_size = best_hyp['batch_size']
            CNN_w.lr = best_hyp['lr']
            CNN_w.krnl_2 = best_hyp['krnl_2']
            CNN_w.mx_krnl_2 = best_hyp['mx_krnl_2']
            tr_acc, tes_acc = CNN_w.trainer()
            tr_data_ave = CNN_w.tr_data_ave
            tr_data_std = CNN_w.tr_data_std
            tes_data_ave = CNN_w.tes_data_ave
            tes_data_std = CNN_w.tes_data_std
            CNN_w.save_model()
            hyp_opt_time = time.time() - st_time
            row = ["Hypeopt", CNN_w.im_size, CNN_w.batch_size, CNN_w.lr, CNN_w.krnl_2, CNN_w.num_epochs,
                   tr_data_ave, tr_data_std, tr_acc, "",
                   tes_data_ave, tes_data_std, tes_acc,
                   hyp_opt_time]

        tmp = 'Best hyps tr acc {} and tes acc {} with tr ave {}, tr std {},' \
              'tes ave {}, tes std {}'.format(tr_acc, tes_acc, tr_data_ave,
                                              tr_data_std, tes_data_ave, tes_data_std)
        print(tmp)
        out_file.write(tmp)
        print("Hypeopt is done in {} sec\n".format(hyp_opt_time))
        writer_f_all.writerow(row)

    elif hype_given:
        #TODO: mnist
        test_size = 1000
        # load_model_name = "hyps.pth"
        load_model_name = None

        out_file = open("res/result_" + str(test_precs) + "_" + model_name + "_givenHyp" +
                 "_epo" + str(num_epoch) + ".txt", "a")
        out_file.write(model_name + "\n")

        f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
        writer_f_all = csv.writer(f_all)

        tr_dir = images_dir + "_" + str(test_precs) + "/tr/"

        st_time = time.time()

        im_size = 64
        batch_size = 234 #231 #50  # [50 - 400]
        lr =  0.00349753559529169 #0.00980093693194154 # 0.0001  # [1e-4, 1e-2]
        krnl_1 = 5  # [3, 10]
        krnl_2 = 8 #1 #5  # [3, 10]
        mx_krnl_1 = 2  # [2, 4]
        mx_krnl_2 =  6 #7 # 2  # [2, 8]
        num_epochs = 1  # [5, 20]
        tmp  = "given hyp started with batch_size {} , lr {} , " \
               "krnl_2 {} , mx_krnl_2 {} with num_epochs {}\n".format(batch_size, lr, krnl_2, mx_krnl_2, num_epochs)
        print(tmp)
        out_file.write(tmp)

        CNN_w = ConvNN_t.CNN_wrap(im_size, batch_size, lr, krnl_1, krnl_2, mx_krnl_1,
                                  mx_krnl_2, num_epochs, tr_dir + "/", tes_dir)
        CNN_w.train_reader()
        CNN_w.test_reader()

        if load_model_name is not None:
            CNN_w.load_model(load_model_name)
            tr_acc = CNN_w.eval_on_tr()
            tes_acc = CNN_w.eval_on_test()
            tr_data_ave = CNN_w.tr_data_ave
            tr_data_std = CNN_w.tr_data_std
            tes_data_ave = CNN_w.tes_data_ave
            tes_data_std = CNN_w.tes_data_std
            tmp = 'Given hyps loaded tr acc {} and tes acc {} with tr ave {}, tr std {},' \
                  'tes ave {}, tes std {}'.format(tr_acc, tes_acc, tr_data_ave,
                                                  tr_data_std, tes_data_ave, tes_data_std)
            print(tmp)
            out_file.write(tmp)

            row = ["Given hyps loaded", CNN_w.im_size, CNN_w.batch_size, CNN_w.lr, CNN_w.krnl_2, CNN_w.num_epochs,
                   tr_data_ave, tr_data_std, tr_acc, "",
                   tes_data_ave, tes_data_std, tes_acc]
            writer_f_all.writerow(row)
        else:
            print("train started on {}\n".format(tr_dir))
            tr_acc, tes_acc = CNN_w.trainer()
            tr_data_ave = CNN_w.tr_data_ave
            tr_data_std = CNN_w.tr_data_std
            tes_data_ave = CNN_w.tes_data_ave
            tes_data_std = CNN_w.tes_data_std
            tmp = 'Given hyps tr acc {} and tes acc {} with tr ave {}, tr std {},' \
                  'tes ave {}, tes std {}'.format(tr_acc, tes_acc, tr_data_ave,
                                                  tr_data_std, tes_data_ave, tes_data_std)
            print(tmp)
            out_file.write(tmp)

            row = ["Given hyps", CNN_w.im_size, CNN_w.batch_size, CNN_w.lr, CNN_w.krnl_2, CNN_w.num_epochs,
                   tr_data_ave, tr_data_std, tr_acc, "",
                   tes_data_ave, tes_data_std, tes_acc]
            writer_f_all.writerow(row)
            CNN_w.save_model()

        #

        df = pd.read_csv(test_precs_file)
        test_size = test_size
        print("test on diff day precs started on {}\n".format(tes_dir))

        for cntr, day_prec in enumerate(df['day prec']):
            num_classes = 0
            for class_fldr in os.listdir(tes_dir):
                if (class_fldr == "FRAMESB"):
                    prec = day_prec
                elif (class_fldr == "FRAMESA"):
                    prec = 1 - day_prec
                else:
                    continue
                num_classes += 1
                images = os.listdir(tes_dir + class_fldr)
                ln_ = len(images)
                random.shuffle(images)
                ln_ = min(int(prec * test_size), ln_)
                print("{} images picked for test\n".format(ln_))
                images_ = images[:ln_]
                tmp_tes = "tmp_test"
                make_dir(tmp_tes)
                make_dir(tmp_tes+"/"+class_fldr)
                for image in images_:
                    copyfile(tes_dir+class_fldr +"/"+ image, tmp_tes+"/"+class_fldr+"/"+ image)

            CNN_w.batch_size = 100
            CNN_w.tes_dir = tmp_tes
            CNN_w.test_reader()
            tes_acc = CNN_w.eval_on_test()
            tes_data_ave = CNN_w.tes_data_ave
            tes_data_std = CNN_w.tes_data_std

            tmp = "tes acc on day prec {} with tes_data_ave {} and tes_data_std {} " \
                  "is {} \n".format(day_prec, tes_data_ave, tes_data_std, tes_acc)
            print(tmp)
            out_file.write(str(tmp))

            row = [prec, CNN_w.im_size, CNN_w.batch_size, CNN_w.lr, CNN_w.krnl_2, CNN_w.num_epochs,
                   tr_data_ave, tr_data_std, tr_acc, "",
                   tes_data_ave, tes_data_std, tes_acc, "small test"]
            writer_f_all.writerow(row)
            os.system("rm -r {}".format(tmp_tes))

        print("Given hyps takes {}".format(time.time()-st_time))

    elif RSO_use:

        df = pd.read_csv(points_list_file)
        # X_test, y_test, num_classes = read_files(tes_dir, model_name)

        if mnist_on:
            for k in ['batch_size', 'fc_size', 'mxp_krnl']:
                df[k] = df[k].astype(int)
            hyp_rngs = {'lr': (1e-4, 1e-1), 'batch_size': (10, 64), 'fc_size': (30, 200), 'mxp_krnl': (2, 10)}
            mymnistTmp = mymnist(hyp_rngs=hyp_rngs)
            mymnistTmp.load_dataset(tr_ss=mnist_tr_size, tes_ss=mnist_tes_size)
            for exp_point in df.iterrows():
                st_time = time.time()
                mymnistTmp.change_blur(blur_prec=exp_point[1]['blur_prec'])
                mymnistTmp.evaluate_model(hyps=dict(exp_point[1]))
                tes_acc = mymnistTmp.evaluate_model(hyps=exp_point[1])
                tr_acc = mymnistTmp.tr_eval()
                tr_data_ave, tr_data_std = mymnistTmp.tr_ave, mymnistTmp.tr_std
                tes_data_ave, tes_data_std = mymnistTmp.tes_ave, mymnistTmp.tes_std

                tmp = 'tr acc {} and tes acc {} on division {} with {} tr ave, {} tr std,' \
                      'tes ave {}, tes std {}'.format(tr_acc, tes_acc, exp_point[0], tr_data_ave,
                                                      tr_data_std, tes_data_ave, tes_data_std)
                print(tmp)
                div_time = time.time() - st_time
                row = [exp_point[0], mymnistTmp.img_size] + list(mymnistTmp.hyps.values)+\
                      [tr_data_ave, tr_data_std, tr_acc, mymnistTmp.trainX.shape, "",
                       tes_data_ave, tes_data_std, tes_acc, mymnistTmp.testX.shape,
                       div_time]
                rowTitle = ['division_num', 'mymnistTmp.img_size'] + list(mymnistTmp.hyps.keys())+\
                      ['tr_data_ave', 'tr_data_std', 'tr_acc', 'trainX.shape', "",
                       'tes_data_ave', 'tes_data_std', 'tes_acc', 'testX.shape',
                       'div_time']
                write_csv(rowTitle, row, file_name='mnist_rso')
        else:
            out_file = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".txt", "a")
            out_file.write(model_name + "\n")

            f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
            writer_f_all = csv.writer(f_all)

            test_dataset = None
            test_load = None
            division_num = 0
            list_dir = os.listdir(divide_files_dir)
            list_dir.sort(key=int)
            for tr_dir in list_dir:
                # import IPython
                # IPython.embed()
                if tr_dir == "res" or division_num > len(df['day prec']):
                    continue
                # print(tr_dir)
                st_time = time.time()
                # read tr
                # X_train, y_train, num_classes = read_files(divide_files_dir+tr_dir+"/", model_name)

                im_size = 64
                batch_size = int(df['batch_size'][division_num])  # [50 - 400]
                lr = float(df['lr'][division_num])  # 0.0001 # [1e-4, 1e-2]
                krnl_1 = 5  # [2, 40]
                krnl_2 = int(df['krnl_1'][division_num])  # 5 # [2, 40]
                mx_krnl_1 = 2  # [2, 4]
                mx_krnl_2 = int(df['mx_krnl_1'][division_num])  # 2 # [2, 8]
                # num_epochs = int(df['num_epochs'][division_num]) # 2 # [5, 40]
                num_epochs = 1
                CNN_w = ConvNN_t.CNN_wrap(im_size, batch_size, lr, krnl_1, krnl_2, mx_krnl_1,
                                          mx_krnl_2, num_epochs, divide_files_dir + tr_dir + "/", tes_dir)
                CNN_w.train_reader()
                if test_dataset is not None:
                    CNN_w.test_dataset = test_dataset
                    CNN_w.test_load = test_load
                    print("used previous test read from {}".format(tes_dir))
                else:
                    CNN_w.test_reader()
                    test_dataset = CNN_w.test_dataset
                    test_load = CNN_w.test_load
                    print("test loaded form {}".format(tes_dir))
                # CNN_w.test_reader()

                print("train started on division {} in {}".format(division_num, divide_files_dir + tr_dir + "/"))
                tr_acc, tes_acc = CNN_w.trainer()
                tr_data_ave = CNN_w.tr_data_ave
                tr_data_std = CNN_w.tr_data_std
                tes_data_ave = CNN_w.tes_data_ave
                tes_data_std = CNN_w.tes_data_std

                if model_name == 'svm':
                    svm = svm_run(X_train, y_train)

                # tr acc
                # tr_acc, tr_prec, tr_reca, tr_f1 = eval(divide_files_dir, division_num, test_precs, model_name,
                #      X_train, y_train, net, svm, f, tr_=True)

                # tmp = 'Acc, prec, recal, f1 on tr '+ str(division_num)+\
                #       " are {}, {}, {}, {} \n".format(tr_acc, tr_prec, tr_reca, tr_f1)
                # print(tmp)
                # f.write(tmp)

                # X_train, y_train = None, None

                tmp = 'tr acc {} and tes acc {} on division {} with {} tr ave, {} tr std,' \
                      'tes ave {}, tes std {}'.format(tr_acc, tes_acc, division_num, tr_data_ave,
                                                      tr_data_std, tes_data_ave, tes_data_std)
                print(tmp)
                out_file.write(tmp)

                # tes acc
                # tes_acc, tes_prec, tes_reca, tes_f1 = eval(divide_files_dir, division_num, test_precs, model_name,
                #      X_test, y_test, net, svm, f, tr_=False)
                #
                # tmp = 'Acc, prec, recal, f1 on tes are {}, {}, {}, {} \n'.format(tes_acc, tes_prec, tes_reca, tes_f1)
                # print(tmp)
                # f.write(tmp)
                div_time = time.time() - st_time
                print("division {} is done in {}\n".format(division_num, div_time))
                # row = [division_num, hidden_dim,learningrate_,lrdecay_,weightdecay_,data_ave,data_std,
                #        tr_acc, tr_prec, tr_reca, tr_f1, "",tes_acc, tes_prec, tes_reca, tes_f1, "",
                #        bias_,hiddenclass_,outclass_,num_epoch,momentum_,batchlearning_]
                row = [division_num, im_size, batch_size, lr, krnl_2, num_epochs,
                       tr_data_ave, tr_data_std, tr_acc, "",
                       tes_data_ave, tes_data_std, tes_acc,
                       div_time]
                writer_f_all.writerow(row)

                division_num += 1

            out_file.close()
