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
from scipy import ndimage, misc
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

np.random.seed(110)

class mymnist():
	def __init__(self, hyp_rngs, img_size=(28, 28), blur_prec=.5):
		self._seed = 110
		np.random.seed(self._seed)
		if not hyp_rngs:
			from init import hyp_rngs
			self.hyp_rngs = hyp_rngs
		else:
			self.hyp_rngs = hyp_rngs

		self.img_size = img_size
		self.blur_prec = blur_prec
		self.rs = np.random.RandomState(seed=self._seed)
		print(f"hyp_rngs {hyp_rngs}\n")

	def load_dataset(self, tr_ss=100, tes_ss=30):
		# load train and test dataset
		# load dataset
		(self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
		# sample
		tr_idx = np.random.randint(0, self.trainX.shape[0], tr_ss)
		tes_idx = np.random.randint(0, self.testX.shape[0], tes_ss)
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
		self.trainX, self.testX, self.trainY, self.testY = \
			self.prep_pixels(self.trainX, self.testX, self.trainY, self.testY)
		# save data
		self.trainX_save, self.testX_save = self.trainX.copy(), self.testX.copy()
		self.trainY_save, self.testY_save = self.trainY.copy(), self.testY.copy()
		# add noisy
		self.trainX, self.trainY = self.add_noisy(self.trainX, self.trainY)
		# cal the data stats
		self.tr_ave, self.tr_std = np.average(self.trainX), np.std(self.trainX)
		self.tes_ave, self.tes_std = np.average(self.testX), np.std(self.testX)
		print(self.tr_ave, self.tr_std, self.tes_ave, self.tes_std)

	def add_noisy(self, x: np.ndarray, y: np.ndarray):
		# rotate
		# self.deg = self.blur_prec * 90
		# self.deg = self.blur_prec * 10
		# X = ndimage.rotate(X, self.deg, reshape=True)
		# X = ndimage.interpolation.rotate(X, self.deg, reshape=False)

		# shift
		# X = ndimage.interpolation.shift(X, self.deg, mode='nearest')

		# blurr
		# X = gaussian_filter(X, sigma=3*self.blur_prec)

		# Add blurries
		bsize = int(x.shape[0]*self.blur_prec)
		x_noise = x[:bsize] + np.random.normal(scale=5, size=x[:bsize].shape)
		x = np.append(x, x_noise, axis=0)
		y = np.append(y, y[:bsize], axis=0)
		print(f"Blurred size: {x.shape, y.shape}")
		return x, y

	def prep_pixels(self, trainX, testX, trainY, testY):
		# scale pixels
		# convert from integers to floats
		trainX = trainX.astype('float32')
		testX = testX.astype('float32')
		# normalize to range 0-1
		trainX = trainX / 255.0
		testX = testX / 255.0
		# return normalized images + noisy images
		return trainX, testX, trainY, testY

	def change_blur(self, blur_prec):
		self.blur_prec = blur_prec
		# self.trainX, self.testX, self.trainY, self.testY = \
		# 	self.prep_pixels(self.trainX_save, self.testX_save, self.trainY_save, self.testX_save)

		self.trainX, self.trainY = self.add_noisy(self.trainX_save, self.trainY_save)

		self.tr_ave, self.tr_std = np.average(self.trainX), np.std(self.trainX)
		self.tes_ave, self.tes_std = np.average(self.testX), np.std(self.testX)

		print(self.tr_ave, self.tr_std, self.tes_ave, self.tes_std)

	def define_model(self, lr=0.01, fc_size=100, mxp_krnl=2):
		# define cnn model
		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.model.add(MaxPooling2D((mxp_krnl, mxp_krnl)))
		self.model.add(Flatten())
		self.model.add(Dense(fc_size, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(10, activation='softmax'))
		# compile model
		opt = SGD(lr=lr, momentum=0.9)
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return self.model

	def run_model(self, trainX, trainY, testX, testY):
		# fit & run for CV
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

	def tr_eval(self):
		_, tr_acc = self.model.evaluate(self.trainX, self.trainY, verbose=0)
		# print('> %.5f' % (tr_acc * 100.0))
		return tr_acc

	def check_hyp(self, hyp_name):
		assert self.hyp_rngs[hyp_name][0] <= self.hyps[hyp_name] <= self.hyp_rngs[hyp_name][1], \
			f"{hyp_name} out of ranges"
		return self.hyps[hyp_name]

	def evaluate_model(self, hyps):
		print(f"hyps {hyps}")
		self.hyps = hyps

		epochs = 10
		batch_size = self.check_hyp('batch_size')
		lr = self.check_hyp('lr')
		fc_size = self.check_hyp('fc_size')
		mxp_krnl = self.check_hyp('mxp_krnl')

		# define model
		self.model = self.define_model(lr=lr, fc_size=int(fc_size), mxp_krnl=int(mxp_krnl))
		# fit model
		history = self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=int(batch_size),
							validation_data=(self.testX, self.testY), verbose=0)
		# evaluate model
		_, tes_acc = self.model.evaluate(self.testX, self.testY, verbose=0)
		print('Test acc: %.5f' % (tes_acc * 100.0))
		return tes_acc


if __name__ == "__main__":

	from init import hyp_rngs

	hyps = {'lr': .01, 'batch_size': 10, 'fc_size': 50, 'mxp_krnl': 2}

	fo = mymnist(hyp_rngs=hyp_rngs, blur_prec=.1)
	# fo.run_test_harness()
	# fo.load_dataset(tr_ss=1000, tes_ss=300)
	fo.load_dataset(tr_ss=10000, tes_ss=3000)
	fo.evaluate_model(hyps)
