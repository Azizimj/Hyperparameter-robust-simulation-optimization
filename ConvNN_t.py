import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
np.random.seed(110)
import time
torch.manual_seed(110)

class CNN_wrap():
    def __init__(self, im_size, batch_size, lr, krnl_1, krnl_2, mx_krnl_1, mx_krnl_2, num_epochs, tr_dir, tes_dir):
        self.im_size = im_size
        self.lr = lr
        self.krnl_1 = krnl_1
        self.krnl_2 = krnl_2
        self.mx_krnl_1 = mx_krnl_1
        self.mx_krnl_2 = mx_krnl_2
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # Transformation for image
        self.transform_ori = transforms.Compose([transforms.RandomResizedCrop(self.im_size ),  # create 64x64 image
                                            transforms.RandomHorizontalFlip(),  # flipping the image horizontally
                                            transforms.ToTensor(),  # convert the image to a Tensor
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])  # normalize the image
        self.tr_dir = tr_dir
        self.tes_dir = tes_dir
        # self.f = open("CNN_"+str(batch_size)+"_"+str(lr)+str(batch_size)+"_"+.txt", "a")
        # self.f.write(model_name + "\n")
        self.CUDA = torch.cuda.is_available()

    def train_reader(self):
        # Load our dataset
        print("reading train set form {}".format(self.tr_dir))
        self.train_dataset = datasets.ImageFolder(root=self.tr_dir, transform=self.transform_ori)
        self.train_load = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      pin_memory = self.CUDA)  # Shuffle to create a mixed batches of 100 of cat & dog images
        # # Show a batch of images
        # def imshow(img):
        #     img = img / 2 + 0.5  # unnormalize
        #     npimg = img.numpy()
        #     plt.figure(figsize=(20, 20))
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))

        # get some random training images
        # dataiter = iter(train_load)
        # images, labels = dataiter.next()
        #
        # # show images
        # imshow(torchvision.utils.make_grid(images))
        # tmp = 1
        #
        # train_load = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                          batch_size=tmp,
        #                                          shuffle=True)  # Shuffle to create a mixed batches of 100 of cat & dog images
        #
        # test_load = torch.utils.data.DataLoader(dataset=test_dataset,
        #                                         batch_size=tmp,
        #                                         shuffle=False)

        # data_ave = np.average([i[0].mean() for i in test_load])
        # data_sq = np.sum([(i[0]**2).sum() for i in test_load])
        # data_std = np.sqrt()

        # Make the dataset iterable

    def test_reader(self):
        print("reading test set form {}".format(self.tes_dir))
        self.test_dataset = datasets.ImageFolder(root=self.tes_dir, transform=self.transform_ori)
        self.test_load = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     pin_memory = self.CUDA)

    def trainer(self):
        self.model = CNN(self.im_size, self.krnl_1, self.krnl_2, self.mx_krnl_1, self.mx_krnl_2)

        if self.CUDA:
            self.model = self.model.cuda()
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training the CNN
        # % % time

        # Define the lists to store the results of loss and accuracy
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        tr_data_ave = torch.tensor(0)
        tr_data_std = torch.tensor(0)
        tes_data_ave = torch.tensor(0)
        tes_data_std = torch.tensor(0)

        # Training
        for epoch in range(self.num_epochs):
            # Reset these below variables to 0 at the begining of every epoch
            start = time.time()
            correct = 0
            iterations = 0
            iter_loss = 0.0

            self.model.train()  # Put the network into training mode

            for i, (inputs, labels) in enumerate(self.train_load):
                # Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)
                # If we have GPU, shift the data to GPU
                # CUDA = torch.cuda.is_available()
                if self.CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
                outputs = self.model(inputs)
                if epoch == 0:
                    tr_data_ave = tr_data_ave + inputs.mean()
                    tr_data_std = tr_data_std + inputs.std()
                loss = loss_fn(outputs, labels)
                # iter_loss += loss.data[0]  # Accumulate the loss
                iter_loss += loss.item()  # Accumulate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                # tp = ((predicted == labels) * (labels == 1)).sum()  # 1 is day
                # tn = ((predicted == labels) * (labels == 0)).sum()
                # fp = ((predicted != labels) * (labels == 0)).sum()
                # fn = ((predicted != labels) * (labels == 1)).sum()
                iterations += 1

            # Record the training loss
            train_loss.append(iter_loss / iterations)
            # Record the training accuracy
            train_accuracy.append((100*correct / len(self.train_dataset)))

            # Testing
            loss = 0.0
            correct = 0
            iterations = 0

            self.model.eval()  # Put the network into evaluation mode

            for i, (inputs, labels) in enumerate(self.test_load):

                # Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                # CUDA = torch.cuda.is_available()
                if self.CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self.model(inputs)
                if epoch == 0:
                    tes_data_ave = tes_data_ave + inputs.mean()
                    tes_data_std = tes_data_std + inputs.std()
                loss = loss_fn(outputs, labels)  # Calculate the loss
                # loss += loss.data[0]
                loss += loss.item()
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()


                iterations += 1

            # Record the Testing loss
            test_loss.append(loss / iterations)
            # Record the Testing accuracy
            test_accuracy.append((100 * correct / len(self.test_dataset)))
            stop = time.time()

            print(
                'Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
                    .format(epoch + 1, self.num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1],
                            stop - start))

        self.tr_data_ave = (tr_data_ave / len(self.train_dataset)).item()
        self.tr_data_std = (tr_data_std / len(self.train_dataset)).item()
        self.tes_data_ave = (tes_data_ave / len(self.test_dataset)).item()
        self.tes_data_std = (tes_data_std / len(self.test_dataset)).item()

        return train_accuracy[-1].item(), test_accuracy[-1].item()

    def plotter(self):
        # Loss
        f = plt.figure(figsize=(10, 10))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(test_loss, label='Testing Loss')
        plt.legend()
        plt.show()

        # Accuracy
        f = plt.figure(figsize=(10, 10))
        plt.plot(train_accuracy, label='Training Accuracy')
        plt.plot(test_accuracy, label='Testing Accuracy')
        plt.legend()
        plt.show()

        # Run this if you want to save the model
        # torch.save(model.state_dict(),'Cats-Dogs.pth')
        #
        # #Run this if you want to load the model
        # model.load_state_dict(torch.load('Cats-Dogs.pth'))

    def predict(self, img_name):
        image = cv2.imread(img_name)  # Read the image
        img = Image.fromarray(image)  # Convert the image to an array
        img = transforms_photo(img)  # Apply the transformations
        img = img.view(1, 3, self.im_size , self.im_size )  # Add batch size
        img = Variable(img)
        # Wrap the tensor to a variable

        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            img = img.cuda()

        output = self.model(img)
        print(output)
        print(output.data)
        _, predicted = torch.max(output, 1)
        if predicted.item() == 0:
            p = 'Cat'
        else:
            p = 'Dog'
        cv2.imshow('Original', image)
        return p

        # pred = predict("F:/Acad/research/fafar/RSO/nd_code/alderley/images_[500,550]_[0.2,0.2]/tr/", model)
        # print("The Predicted Label is {}".format(pred))


class CNN(nn.Module):
    def __init__(self, im_size, krnl_1, krnl_2, mx_krnl_1, mx_krnl_2):
        super(CNN, self).__init__()

        self.im_size = im_size
        self.krnl_1 = krnl_1 # 3
        # self.krnl_2 = krnl_2 # 5
        self.cnn1_out_dim = im_size+2-krnl_1+1  # floor((H_in +2 * Padding -krnl)/stride)+1)
        self.krnl_2 = min(krnl_2, max(self.cnn1_out_dim-10, 1)  ) # to control the krnl size
        self.mx_krnl_1 = mx_krnl_1 #2
        self.mx_krnl_2 = mx_krnl_2 #2
        self.cnn2_out_dim = self.cnn1_out_dim/self.mx_krnl_1+4-self.krnl_2+1

        self.out_cnn1 = 8
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=self.out_cnn1, kernel_size=self.krnl_1, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(self.out_cnn1)  # Batch normalization
        self.relu = nn.ReLU()  # RELU Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.mx_krnl_1)  # Maxpooling reduces the size by kernel size. 64/2 = 32

        self.out_cnn2 = 32

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=self.out_cnn2, kernel_size=self.krnl_2, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(self.out_cnn2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.mx_krnl_2)  # Size now is 32/2 = 16

        # Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is
        # of size 16x16 --> 32*16*16 = 8192
        sq_image_size = int(self.cnn2_out_dim/ self.mx_krnl_2)
        self.in_features_size = int(self.out_cnn2* sq_image_size**2)
        self.fc1 = nn.Linear(in_features=self.in_features_size,
                             out_features=4000)  # Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)  # Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50,
                             out_features=2)  # Since there were so many features,
        # I decided to use 45 layers to get output layers.
        # You can increase the kernels in Maxpooling to reduce image further and reduce number
        # of hidden linear layers.

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out) #
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        # Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1, self.in_features_size)  # -1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        # Then we forward through our fully connected layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out

