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
##hyp
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
##
import csv

# import tensorflow as tf

import torch.nn as nn
import torch.nn.functional as F

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


class nn_hold():
    def __init__(self, bias_, hiddenclass_, outclass_, momentum_, batchlearning_ ):
        self.bias_ = bias_
        self.hiddenclass_ = hiddenclass_
        self.outclass_ = outclass_
        self.momentum_ = momentum_
        self.batchlearning_ = batchlearning_

    def nn_run(self, hidden_dim, num_epoch, learningrate_, lrdecay_, weightdecay_, num_classes, X_train, y_train):
        # # NN HYP
        # # hidden_dim = 100
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
        # batchlearning_ = False
        # # weightdecay_ = 0.01
        # # NN HYP

        net = buildNetwork(SHAPE[0] * SHAPE[1] * 3, hidden_dim,
                           num_classes, bias=self.bias_, hiddenclass=self.hiddenclass_, outclass=self.outclass_)

        train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)
        test_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)

        if batch_size ==0:
            # for feature, label in zip(X_train, y_train):
            for feature, label in zip(X_train, y_train):
                train_ds.addSample(feature, label)

            # for feature, label in zip(X_test, y_test):
            #     test_ds.addSample(feature, label)

            # checking for model
            if os.path.isfile("models/" + model_name + ".pkl"):
                tmp = "Using previous " + model_name + " model...\n"
                print(tmp)
                f.write(tmp)
                trainer = pickle.load(open("models/" + model_name + ".pkl", "rb"))
            else:
                # tmp = "Training " + model_name + " on set " + str(division_num) + "\n"
                # print(tmp)
                # f.write(tmp)
                trainer = BackpropTrainer(net, train_ds, learningrate=learningrate_, lrdecay=lrdecay_,
                                          momentum=self.momentum_, verbose=True, batchlearning=self.batchlearning_,
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
        elif batch_size>0:
            trainer = BackpropTrainer(net, learningrate=learningrate_, lrdecay=lrdecay_,
                                      momentum=self.momentum_, verbose=True,
                                      batchlearning=self.batchlearning_,
                                      weightdecay=weightdecay_)
            for epoch in range(num_epoch):
                print("\n epoch {}".format(epoch))
                for i in range(X_train.shape[0] // batch_size):
                    X_ = X_train[i * batch_size:(i + 1) * batch_size][:]
                    y_ = y_train[i * batch_size:(i + 1) * batch_size]

                    tmp = "epoch {}, batch {}".format(epoch, i)
                    print(tmp)
                    f.write(tmp)

                    train_ds = SupervisedDataSet(SHAPE[0] * SHAPE[1] * 3, num_classes)

                    for feature, label in zip(X_, y_):
                        train_ds.addSample(feature, label)

                    # train_ds.batches("batches", batch_size)

                    # for feature, label in zip(X_test, y_test):
                    #     test_ds.addSample(feature, label)

                    # checking for model
                    if os.path.isfile("models/" + model_name + ".pkl"):
                        tmp = "Using previous " + model_name + " model...\n"
                        print(tmp)
                        f.write(tmp)
                        trainer = pickle.load(open("models/" + model_name + ".pkl", "rb"))
                    else:
                        # tmp = "Training " + model_name + " on set " + str(division_num) + "\n"
                        # print(tmp)
                        # f.write(tmp)
                        # trainer = BackpropTrainer(net, learningrate=learningrate_, lrdecay=lrdecay_,
                        #                           momentum=self.momentum_, verbose=True,
                        #                           batchlearning=self.batchlearning_,
                        #                           weightdecay=weightdecay_)
                        # different trainig calls
                        # trainer.train()
                        trainer.trainOnDataset(train_ds)
                        # trainer.trainOnDataset(dataset)
                        # trainer.trainUntilConvergence(dataset=None, maxEpochs=None,
                        #                               verbose=None, continueEpochs=10, validationProportion=0.25)
                        # different trainig calls

                        # print("Saving model")
                        # pickle.dump(trainer, open("models/"+ model_name+ ".pkl", "wb"))

                    # tmp = eval(" ", " ", test_precs, model_name,
                    #            X_train, y_train, net, svm, f, tr_=True)
                    # print("eval {}".format(tmp))

        return net


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    def traner(self):
        # net = Net()
        # print(net)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # create a loss function
        criterion = nn.NLLLoss()

        # run the main training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                data = data.view(-1, 28*28)
                optimizer.zero_grad()
                net_out = self.forward(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.data[0]))

    def pred(self):
        # run a test loop
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            data = data.view(-1, 28 * 28)
            net_out = net(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data[0]
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def opt(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def trainer(self):
        # Train the model
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Run the forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))

    def tester(self):
        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

        # Save the model and plot
        torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')


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

    eval_ = eval(tr_dir, "hpopt", test_precs, model_name, X_train, y_train, net, svm, f, tr_=True)

    return eval_[0] # acc


if __name__ == "__main__":

    tr_tes_sep = False
    sample_folder_build = False
    divide_file = False
    hyperopt_use = False
    hype_given = True
    RSO_use = False
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

    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
    images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
    # images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images_[500,550]"
    # images_dir = "images"

    make_dir("res/")

    size_of_trs = 6000
    # size_of_trs = 50

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

    # hyps:
    # int:
    # hidden_dim[20, 200]
    hidden_dim_l = 20
    hidden_dim_u = 200
    #
    # Real:
    # learningrate_[1e-5, 0.1]
    learningrate_l = 1e-5
    learningrate_u = 0.1
    # lrdecay_[1e-2, 1e-1]
    lrdecay_l = 1e-2
    lrdecay_u = 1e-1
    # weightdecay_[1e-3, 0.9]
    weightdecay_l = 1e-3
    weightdecay_u = 0.9

    points_list_file = "Design-Data.csv"
    # points_list_file = "Design-Data-small.csv"
    test_precs_file = "Test-Data.csv"

    # FRAMESA (night) 16960, FRAMESB (day) 14607
    classes_labels = ["FRAMESA", "FRAMESB"]
    # sample_sizes = [100, 200]
    sample_sizes = [500, 550]
    # sample_sizes = [-1,-1] # -1 for not sampling
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

    # read_files("F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]_[0.2, 0.2]/tr/", model_name)
    # exit()

    tes_dir = images_dir + "_" + str(test_precs) + "/"+"tes/"

    divide_files_dir = images_dir +"_"+ str(test_precs) + "/tr/"+"divided/"
    division_num = 0

    # NN HYP
    # hidden_dim = 100
    bias_ = True
    # SoftmaxLayer, TanhLayer, SigmoidLayer, LSTMLayer, LinearLayer, GaussianLayer
    hiddenclass_ = TanhLayer
    outclass_ = SoftmaxLayer
    # num_epoch = 4
    # if len(sys.argv)>0:
    #     num_epoch = int(sys.argv[1])
    # learningrate_ = 0.01
    # lrdecay_ = 1.0
    momentum_ = 0.1
    batchlearning_ = True
    # weightdecay_ = 0.01
    # NN HYP

    #hypopt
    if hyperopt_use:
        max_eval_hpopt = 2
        if len(sys.argv) > 1:
            max_eval_hpopt = int(sys.argv[3])

        f = open("res/result_" + str(test_precs) + "_" + model_name + "_hyopt"+str(max_eval_hpopt)+
                 "_epo" + str(num_epoch) + ".txt", "a")
        f.write(model_name + "\n")

        f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
        writer_f_all = csv.writer(f_all)

        tr_dir = images_dir + "_" + str(test_precs) + "/tr/"
        X_train, y_train, num_classes = read_files(tr_dir, model_name)

        print("hyp started ")

        nn_hold_ = nn_hold(bias_, hiddenclass_, outclass_, momentum_, batchlearning_)

        # hyps = {'hidden_dim':5, 'learningrate_':0.01, 'lrdecay_': 1.0, 'weightdecay_': 0.01}
        # obj = objective_(hyps=hyps)
        # print(obj)

        # define a search space
        # 'hidden_dim': hp.uniform('hidden_dim', hidden_dim_l, hidden_dim_u),
        space_ = {'hidden_dim': hp.randint('hidden_dim', hidden_dim_u-hidden_dim_l+1),
                  'learningrate_': hp.uniform('learningrate_', learningrate_l, learningrate_u ),
                  'lrdecay_': hp.uniform('lrdecay_', lrdecay_l, lrdecay_u),
                  'weightdecay_': hp.uniform('weightdecay_', weightdecay_l, weightdecay_u)}

        # minimize the objective over the space
        best_hyp = fmin(objective_, space_, algo=tpe.suggest, max_evals=max_eval_hpopt)

        tmp = "\n optimal hyps with tpe hypopt are {}\n".format(best_hyp)
        f.write(tmp)
        print(tmp)
        # tmp = "space eval at optimal hyps is {}\n".format(space_eval(space_, best_hyp))
        # f.write(tmp)
        # print(tmp)

        hidden_dim = best_hyp['hidden_dim']
        learningrate_ = best_hyp['learningrate_']
        lrdecay_ = best_hyp['lrdecay_']
        weightdecay_ = best_hyp['weightdecay_']
        net = nn_hold_.nn_run(hidden_dim, num_epoch, learningrate_, lrdecay_,
                              weightdecay_, num_classes, X_train, y_train)

        tr_acc, tr_prec, tr_reca, tr_f1 = eval(" ", " ", test_precs, model_name,
                                               X_train, y_train, net, svm, f, tr_=True)

        tmp = 'Acc, prec, recal, f1 on tr for Hyperopt are ' \
              '{}, {}, {}, {} \n'.format(tr_acc, tr_prec, tr_reca, tr_f1)
        print(tmp)
        f.write(tmp)

        X_test, y_test, num_classes = read_files(tes_dir, model_name)
        tes_acc, tes_prec, tes_reca, tes_f1 = eval(" ", " ", test_precs, model_name,
                        X_test, y_test, net, svm, f, tr_=False)

        tmp = 'Acc, prec, recal, f1 of BEST hyp on tes are {}\n'.format(tes_eval)
        print(tmp)
        f.write(tmp)

        if model_name == "nn":
            row = ["hyperopt", hidden_dim, learningrate_,
                   lrdecay_, weightdecay_, data_ave, data_std,
                   tr_acc, tr_prec, tr_reca, tr_f1, "", tes_acc, tes_prec, tes_reca, tes_f1, "",
                   bias_, hiddenclass_, outclass_, num_epoch, momentum_, batchlearning_]
            writer_f_all.writerow(row)

    elif hype_given:
        hidden_dim = 134
        learningrate_ = 0.0467821978480138
        # learningrate_ = 0.0001
        lrdecay_ = 0.0742889052103925
        weightdecay_ = 0.71933869547565

        f = open("res/result_" + str(test_precs) + "_" + model_name + "_givenHyp" +
                 "_epo" + str(num_epoch) + ".txt", "a")
        f.write(model_name + "\n")

        f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
        writer_f_all = csv.writer(f_all)

        tr_dir = images_dir + "_" + str(test_precs) + "/tr/divided/0/"
        X_train, y_train, num_classes = read_files(tr_dir, model_name)

        X_sample = np.concatenate((X_train[:, 500:600], X_train[:, 1100:1200],
                                   X_train[:, 1600:1700], X_train[:, 2000:2100]), axis=1)
        data_ave = np.average(X_sample)
        data_std = np.std(X_train)

        print("given hyp started ")

        nn_hold_ = nn_hold(bias_, hiddenclass_, outclass_, momentum_, batchlearning_)

        print("train started on {}".format(tr_dir))

        net = nn_hold_.nn_run(hidden_dim, num_epoch, learningrate_, lrdecay_,
                              weightdecay_, num_classes, X_train, y_train)

        #train eval on tr
        tr_acc, tr_prec, tr_reca, tr_f1 = eval(" ", " ", test_precs, model_name,
                                               X_train, y_train, net, svm, f, tr_=True)

        tmp = "train tr_acc, tr_prec, tr_reca, tr_f1 " \
              "on the whole train are {}, {}, {}, {} \n".format(tr_acc, tr_prec, tr_reca, tr_f1)
        print(tmp)
        f.write(str(tmp))

        if model_name == "nn":
            row = ["whole tr, hyp given", hidden_dim, learningrate_, lrdecay_, weightdecay_, data_ave, data_std,
                   tr_acc, tr_prec, tr_reca, tr_f1, "", " ", " ", " ", " ", "",
                   bias_, hiddenclass_, outclass_, num_epoch, momentum_, batchlearning_]
            writer_f_all.writerow(row)

        #
        print("test started on {}".format(tes_dir))

        df = pd.read_csv(test_precs_file)
        test_size = 20

        for cntr, day_prec in enumerate(df['day prec']):
            X_test = []
            y_test = []
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
                print("{} images picked".format(ln_))
                images_ = images[:ln_]

                for image in images_:
                    X_test.append(extract_feature(tes_dir + class_fldr+"/"+image))
                    y_test.append(class_fldr)
            if model_name == 'nn':
                y_test = convertLabels(y_test)

            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test)

            X_sample = np.concatenate((X_test[:, 500:600], X_test[:, 1100:1200],
                                       X_test[:, 1600:1700], X_test[:, 2000:2100]), axis=1)
            data_ave = np.average(X_sample)
            data_std = np.std(X_test)

            tes_acc, tes_prec, tes_reca, tes_f1 = eval(" ", "given_hyp", test_precs, model_name,
                            X_test, y_test, net, svm, f, tr_=False)

            tmp = "tes_acc, tes_prec, tes_reca, tes_f1 " \
                  "on day prec {} are {}, {}, {}, {} \n".format(day_prec, tes_acc, tes_prec, tes_reca, tes_f1)
            print(tmp)
            f.write(str(tmp))

            if model_name == "nn":
                row = [day_prec, hidden_dim, learningrate_,
                       lrdecay_, weightdecay_, data_ave, data_std,
                       tr_acc, tr_prec, tr_reca, tr_f1, "", tes_acc, tes_prec, tes_reca, tes_f1, "",
                       bias_, hiddenclass_, outclass_, num_epoch, momentum_, batchlearning_]
                writer_f_all.writerow(row)

    #RSO
    elif RSO_use:

        f = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".txt", "a")
        f.write(model_name + "\n")

        f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
        writer_f_all = csv.writer(f_all)

        df = pd.read_csv(points_list_file)
        X_test, y_test, num_classes = read_files(tes_dir, model_name)
        for tr_dir in os.listdir(divide_files_dir):
            if tr_dir == "res" or division_num>len(df['hidden-dim']):
                continue
            # print(tr_dir)
            # read tr
            X_train, y_train, num_classes = read_files(divide_files_dir+tr_dir+"/", model_name)

            hidden_dim = df['hidden-dim'][division_num]
            learningrate_ = df['learningrate'][division_num]
            lrdecay_ = df['Irdecay'][division_num]
            weightdecay_ = df['weightdecay'][division_num]
            # data_ave = np.average(X_train)
            X_sample = np.concatenate((X_train[:, 500:600], X_train[:, 1100:1200],
                                     X_train[:, 1600:1700], X_train[:, 2000:2100]), axis=1)
            data_ave = np.average(X_sample)
            data_std = np.std(X_train)


            if model_name == 'nn':
                nn_hold_ = nn_hold(bias_, hiddenclass_, outclass_, momentum_, batchlearning_)
                net = nn_hold_.nn_run(hidden_dim, num_epoch, learningrate_, lrdecay_, weightdecay_, num_classes, X_train, y_train)

            elif model_name == 'svm':
                svm = svm_run(X_train, y_train)

            # tr acc
            tr_acc, tr_prec, tr_reca, tr_f1 = eval(divide_files_dir, division_num, test_precs, model_name,
                 X_train, y_train, net, svm, f, tr_=True)

            tmp = 'Acc, prec, recal, f1 on tr '+ str(division_num)+\
                  " are {}, {}, {}, {} \n".format(tr_acc, tr_prec, tr_reca, tr_f1)
            print(tmp)
            f.write(tmp)

            X_train, y_train = None, None

            # tes acc
            tes_acc, tes_prec, tes_reca, tes_f1 = eval(divide_files_dir, division_num, test_precs, model_name,
                 X_test, y_test, net, svm, f, tr_=False)

            tmp = 'Acc, prec, recal, f1 on tes are {}, {}, {}, {} \n'.format(tes_acc, tes_prec, tes_reca, tes_f1)
            print(tmp)
            f.write(tmp)

            if model_name == "nn":
                row = [division_num, hidden_dim,learningrate_,lrdecay_,weightdecay_,data_ave,data_std,
                       tr_acc, tr_prec, tr_reca, tr_f1, "",tes_acc, tes_prec, tes_reca, tes_f1, "",
                       bias_,hiddenclass_,outclass_,num_epoch,momentum_,batchlearning_]
                writer_f_all.writerow(row)

            division_num += 1

    f.close()
