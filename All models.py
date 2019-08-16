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
import pandas as pd
from shutil import copyfile

import csv

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
        print(dir)
        os.mkdir(dir)

def test_train_sep(images_dir, test_precs):
    from shutil import copyfile
    print("test_train_sep folder")
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
    print("Reading files...")
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            # if sample_sizes[0] > 0:
            #     images = sample(images, sample_sizes[num_classes - 1])  # sample
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

def divide_with_prec(points_list, dire, size_of_trs):

    # import pandas as pd
    # from shutil import copyfile
    df = pd.read_csv(points_list)
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
    tmp = "Eval...\n"
    print(tmp)
    f.write(tmp)
    make_dir(divide_files_dir + "res/")
    if tr_:
        pred_file = open(divide_files_dir + "res/tr" + str(division_num) + "_preds"
                         + str(test_precs) + "_" + model_name + ".csv", 'a')
    else:
        pred_file = open(divide_files_dir + "res/tes" + str(division_num) + "_preds"
                         + str(test_precs) + "_" + model_name + ".csv", 'a')

    writer_pred_file = csv.writer(pred_file)

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

        row = [total_count, label, prediction, correct_count]
        writer_pred_file.writerow(row)
        total_count += 1
    acc = float(correct_count) / float(total_count)
    precision = float(tp)/float(tp+fp)
    recall = float(tp)/float(tp+fn)
    f1 = float(2*tp)/float(2*tp+fp+fn)
    tmp = 'Acc, prec, recal, f1 on ' +str("tr " if tr_ else "tes ")+\
          str(division_num)+" are {}, {}, {}, {} \n".format(acc, precision, recall, f1)
    print(tmp)
    f.write(tmp)
    pred_file.close()

    return acc, precision, recall, f1

if __name__ == "__main__":

    tr_tes_sep = False
    sample_folder_build = False
    divide_file = False
    if sys.argv[2]=="tr_tes_sep":
        tr_tes_sep = True
    if sys.argv[2]=="sample_folder_build":
        sample_folder_build = True
    if sys.argv[2]=="divide_file":
        divide_file = True

    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
    images_dir = "images"

    # init net and svm
    net = None
    svm = None
    #
    model_name = 'nn'
    # model_name = 'svm'
    random_state = 12

    points_list_file = "Design-Data.csv"
    # points_list_file = "Design-Data-small.csv"

    size_of_trs = 6000
    # size_of_trs = 50

    # FRAMESA (night) 16960, FRAMESB (day) 14607
    # sample_sizes = [100, 200]
    sample_sizes = [-1,-1] # -1 for not sampling
    test_precs= [.2,.2]
    SHAPE = (30, 30)

    if sample_folder_build:
        # Build a sample folder or seperate test and train
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

    num_epoch = 4
    if len(sys.argv) > 0:
        num_epoch = sys.argv[1]

    tes_dir = images_dir + "_" + str(test_precs) + "/"+"tes/"
    X_test, y_test, num_classes = read_files(tes_dir, model_name)

    divide_files_dir = images_dir +"_"+ str(test_precs) + "/tr/"+"divided/"
    division_num = 0
    df = pd.read_csv(points_list_file)

    f = open("res/result_"+str(test_precs) +"_"+model_name+"_epo"+str(num_epoch)+ ".txt", "a")
    f.write(model_name + "\n")

    f_all = open("res/result_" + str(test_precs) + "_"+model_name+"_epo"+str(num_epoch)+".csv", 'a')
    writer_f_all = csv.writer(f_all)


    for tr_dir in os.listdir(divide_files_dir):
        if tr_dir == "res":
            continue
        # read tr
        X_train, y_train, num_classes = read_files(divide_files_dir+tr_dir+"/", model_name)

        hidden_dim = df['hidden-dim'][division_num]
        learningrate_ = df['learningrate'][division_num]
        lrdecay_ = df['Irdecay'][division_num]
        weightdecay_ = df['weightdecay'][division_num]
        data_ave = np.average(X_train)
        data_std = np.std(X_train)


        if model_name == 'nn':

            # NN HYP
            # hidden_dim = 100
            bias_ = True
            # SoftmaxLayer, TanhLayer, SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
            hiddenclass_ = TanhLayer
            outclass_ = SoftmaxLayer
            # num_epoch = 4
            # if len(sys.argv)>0:
            #     num_epoch = sys.argv[1]
            # learningrate_ = 0.01
            # lrdecay_ = 1.0
            momentum_ = 0.1
            batchlearning_ = False
            # weightdecay_ = 0.01
            # NN HYP

            net = buildNetwork(SHAPE[0] * SHAPE[1] * 3, hidden_dim,
                               num_classes, bias=bias_, hiddenclass=hiddenclass_, outclass=outclass_)

            train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
            test_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)

            for feature, label in zip(X_train, y_train):
                train_ds.addSample(feature, label)

            for feature, label in zip(X_test, y_test):
                test_ds.addSample(feature, label)

            # checking for model
            if os.path.isfile("models/" + model_name + ".pkl"):
                tmp = "Using previous " + model_name + " model...\n"
                print(tmp)
                f.write(tmp)
                trainer = pickle.load(open("models/" + model_name + ".pkl", "rb"))
            else:
                tmp = "Training " + model_name + " on set "+str(division_num)+ "\n"
                print(tmp)
                f.write(tmp)
                trainer = BackpropTrainer(net, train_ds, learningrate=learningrate_, lrdecay=lrdecay_,
                                          momentum=momentum_, verbose=True, batchlearning=batchlearning_,
                                          weightdecay=weightdecay_)
                # different trainig calls
                # trainer.train()
                trainer.trainEpochs(epochs=num_epoch)
                # trainer.trainOnDataset(dataset)
                # trainer.trainUntilConvergence(dataset=None, maxEpochs=None,
                #                               verbose=None, continueEpochs=10, validationProportion=0.25)
                # different trainig calls

                # print("Saving model")
                # pickle.dump(trainer, open("models/"+ model_name+ ".pkl", "wb"))

        elif model_name == 'svm':

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

        # tr acc
        tr_acc, tr_prec, tr_reca, tr_f1 = eval(divide_files_dir, division_num, test_precs, model_name,
             X_train, y_train, net, svm, f, tr_=True)

        # tes acc
        tes_acc, tes_prec, tes_reca, tes_f1  = eval(divide_files_dir, division_num, test_precs, model_name,
             X_train, y_train, net, svm, f, tr_=False)

        if model_name == "nn":
            row = [division_num,hidden_dim,learningrate_,lrdecay_,weightdecay_,data_ave,data_std,
                   tr_acc, tr_prec, tr_reca, tr_f1, "",tes_acc, tes_prec, tes_reca, tes_f1, "",
                   bias_,hiddenclass_,outclass_,num_epoch,momentum_,batchlearning_]
            writer_f_all.writerow(row)

        division_num += 1

    f.close()
