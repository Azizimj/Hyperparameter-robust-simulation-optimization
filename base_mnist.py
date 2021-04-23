# baseline cnn model for mnist
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-
# network-from-scratch-for-mnist-handwritten-digit-classification/
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


class mymnist():
	def __init__(self):
		np.random.seed(110)

	def load_dataset(self, tr_ss=100, tes_ss=30):
		# load train and test dataset
		# load dataset
		(self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
		# sample
		tr_idx, tes_idx = np.random.randint(0, self.trainX.shape[0], tr_ss), \
						  np.random.randint(0, self.testX.shape[0], tes_ss)
		self.trainX, self.trainY = self.trainX[tr_idx, :], self.trainY[tr_idx]
		self.testX, self.testY = self.testX[tes_idx, :], self.testY[tes_idx]
		# reshape dataset to have a single channel
		self.trainX = self.trainX.reshape((self.trainX.shape[0], 28, 28, 1))
		self.testX = self.testX.reshape((self.testX.shape[0], 28, 28, 1))
		# one hot encode target values
		self.trainY = to_categorical(self.trainY)
		self.testY = to_categorical(self.testY)
		print(f'train size:{self.trainX.shape}, test shape {self.testX.shape}')
		# prepare pixel data
		self.trainX, self.testX = self.prep_pixels(train=self.trainX, test=self.testX)

	def prep_pixels(self, train, test):
		# scale pixels
		# convert from integers to floats
		train_norm = train.astype('float32')
		test_norm = test.astype('float32')
		# normalize to range 0-1
		train_norm = train_norm / 255.0
		test_norm = test_norm / 255.0
		# return normalized images
		return train_norm, test_norm

	def define_model(self, lr=0.01, krnl_size=100, max_krnl=2):
		# define cnn model
		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.model.add(MaxPooling2D((max_krnl, max_krnl)))
		self.model.add(Flatten())
		self.model.add(Dense(krnl_size, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=lr, momentum=0.9)
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return self.model

	def run_model(self, trainX, trainY, testX, testY):
		# fit model
		history = self.model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = self.model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))

		return acc, history

	def cv_evaluate_model(self, dataX, dataY, n_folds=5):
		# evaluate a model using k-fold cross-validation
		scores, histories = list(), list()
		# prepare cross validation
		kfold = KFold(n_folds, shuffle=True, random_state=1)
		# enumerate splits
		self.define_model()
		for train_ix, test_ix in kfold.split(dataX):
			acc, history = self.run_model(dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix])
			# stores scores
			scores.append(acc)
			histories.append(history)
		return scores, histories

	def _diagnostics(self, histories):
		# plot diagnostic learning curves

		for i in range(len(histories)):
			# plot loss
			pyplot.subplot(2, 1, 1)
			pyplot.title('Cross Entropy Loss')
			pyplot.plot(histories[i].history['loss'], color='blue', label='train')
			pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
			# plot accuracy
			pyplot.subplot(2, 1, 2)
			pyplot.title('Classification Accuracy')
			pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
			pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
		pyplot.show()

	def _performance(self, scores):
		# summarize model performance

		# print summary
		print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
		# box and whisker plots of results
		# pyplot.boxplot(scores)
		# pyplot.show()
		pass

	def run_test_harness(self):
		# run the test harness for evaluating a model

		# load dataset
		self.load_dataset(tr_ss=100, tes_ss=30)
		# evaluate model
		scores, histories = self.cv_evaluate_model(self.trainX, self.testX)
		# learning curves
		# summarize_diagnostics(histories)
		# summarize estimated performance
		self._performance(scores)

	def evaluate_model(self, lr=0.01, batch_size=32, krnl_size=100, max_krnl=2, epochs=10):
		# define model
		model = self.define_model(lr=lr, krnl_size=krnl_size, max_krnl=max_krnl)
		# fit model
		history = model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=batch_size,
							validation_data=(self.testX, self.testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(self.testX, self.testY, verbose=0)
		print('> %.3f' % (acc * 100.0))

		return acc, history

if __name__ == "__main__":

	fo = mymnist()
	# fo.run_test_harness()
	fo.load_dataset(tr_ss=100, tes_ss=30)
	fo.evaluate_model(lr=0.01, batch_size=32, krnl_size=100, max_krnl=2, epochs=2)