# https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network/blob/master/eeg_lstm-v2.ipynb

# Changelog:

# Apply detrend to eeg columns
# Drop eeg columns that are not named "eeg.{x}"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import signal

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical

###
# Helper functions to preprocess data
###

def detrend_eeg(eeg):
	for (name, col) in eeg.iteritems():
		if not name == "Time":
			eeg[name] = signal.detrend(col)
	
	return eeg

	# return eeg.drop("Time").apply(signal.detrend, axis=0)

def slice_eeg_into_samples(eeg, events, sample_length):
	samples_list = []
	
	# drop columns that are not "eeg.x"
	to_drop = [i for i in eeg.columns if not i[:3] == "eeg" and not i == "Time"]
	eeg = eeg.drop(to_drop, axis=1)

	for index, row in events.iterrows():
		index_range = eeg[eeg["Time"] >= row["latency"]].index[0]	# index of first eeg datapoint that happens after this event
		this_slice = eeg[:index_range + 1]	# take slice of datapoints that happened before this event
		this_slice = this_slice.drop("Time", axis=1)	# drop time column
		
		samples_list.append(this_slice)
		eeg = eeg[index_range:].reset_index(drop=True)	# drop rows that happened before this event

	# samples are not exactly the same length
	# pad on filler entries to make every sample as long as the longest one
	# if sample_length is specified, then just pad every sample to sample_length
	samples_list = pad_samples(samples_list, sample_length)
	
	# the loop leaves out measurements of the last event, since each iteration appends datapoints that occur before this event
	# add it here, specifying number of rows and dropping time column
	samples_list.append(eeg.head(len(samples_list[0].index)).drop("Time", axis=1))

	# the loop includes the first measurements, before any event takes place
	# drop it here
	samples_list.pop(0)

	# samples_list is now a python list of pandas dataframes
	# need to convert to a 3D numpy array
	eeg_array = []
	for sample in samples_list:
		eeg_array.append(sample.to_numpy())
	eeg_array = np.stack(eeg_array, axis=0)
	
	return eeg_array

def pad_samples(eeg_old, sample_length):
	eeg_new = []

	# if sample_length is non-zero, then use it instead of figuring out the length
	longest = 0
	if sample_length == 0:
		for sample in eeg_old:
			l = len(sample.index)
			if l > longest:
				longest = l
	else:
		longest = sample_length
	
	# pad each sample with the last row
	for sample in eeg_old:
		for i in range(longest - len(sample.index)):
			sample = sample.append(sample.iloc[-1], ignore_index=True)
		eeg_new.append(sample)
	
	return eeg_new

def one_hot_events(events):
	events_list = list(events["type"])
	lb = preprocessing.LabelBinarizer()
	lb.fit(events_list)
	events_1hot = lb.transform(events_list)
	return events_1hot, lb

def split_train_test(eeg, events):
	sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
	sss.get_n_splits(eeg, events)

	for train_index, test_index in sss.split(eeg, events):
		X_train, X_test = eeg[train_index], eeg[test_index]
		y_train, y_test = events[train_index], events[test_index]
	
	return X_train, X_test, y_train, y_test

###
# Legacy
###

def old():
	eeg_df = pd.read_csv("eeg_test.csv", sep="\t")
	# eeg_df = eeg_df[["Time", "thumb_near"]]	# drop all columns except Time and thumb_near
	eeg_df = eeg_df.dropna()	# drop all rows without a thumb_near value
	# eeg_df = (eeg_df - eeg_df.min()) / (eeg_df.max() - eeg_df.min())	# normalise all columns to be between 0 and 1

	events_df = pd.read_csv("eeg_events.csv", sep="\t")
	# events_df = events_df[(events_df.type >= 1536) & (events_df.type <= 1542)]	# only keep rows corresponding to hand movement events

###
# Main
###

def load_data():
	time_steps = 2200

	eeg_all = np.zeros(0)
	events_all = np.zeros(0)
	for i in range(1, 11):
		print("starting file {}".format(i))
		eeg_df = pd.read_csv("motorexecution1/subject1_run{}.csv".format(i), sep="\t")
		eeg_df = eeg_df.dropna()
		events_df = pd.read_csv("motorexecution1/subject1_events{}.csv".format(i), sep="\t")
		events_df = events_df[(events_df.type >= 1536) & (events_df.type <= 1542)]	# only keep rows corresponding to hand movement events
		print("read")

		# Apply detrend to eeg columns
		eeg_df = detrend_eeg(eeg_df)

		eeg_samples = slice_eeg_into_samples(eeg_df, events_df, time_steps)
		events_1hot, lb = one_hot_events(events_df)
		print("sliced")

		assert eeg_samples.shape[0] == events_1hot.shape[0], "eeg_samples dim does not match events_1hot"

		if not eeg_all.any():
			eeg_all = eeg_samples
			events_all = events_1hot
		else:
			eeg_all = np.concatenate((eeg_all, eeg_samples), axis=0)
			events_all = np.concatenate((events_all, events_1hot), axis=0)
		
		print("file {} done".format(i))
		print("dimensions:", eeg_all.shape, events_all.shape, "\n")

	assert eeg_all.shape[0] == events_all.shape[0], "eeg_all dim does not match events_all"

	np.save("motorexecution1/eeg_all", eeg_all)
	np.save("motorexecution1/events_all", events_all)

	return eeg_all, events_all, lb

eeg_all, events_all, lb = load_data()
# eeg_all = np.load("motorexecution1/eeg_all.npy")
# events_all = np.load("motorexecution1/events_all.npy")

num_of_samples = eeg_all.shape[0]	# num of samples (hand movement events)
n_features = eeg_all.shape[2]		# num of eeg channels
time_steps = eeg_all.shape[1]		# num of time steps in each sample
event_types = events_all.shape[1]	# num of event types

X_train, X_test, y_train, y_test = split_train_test(eeg_all, events_all)

# Train LSTM model, using options from EEG paper linked at the top
# dense layer for 1_hot: https://stackoverflow.com/questions/49604765/

def train_model():
	model = Sequential()
	model.add(LSTM(100, return_sequences=False, input_shape=(time_steps, n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(event_types, activation="sigmoid"))

	model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

	history = model.fit(X_train, y_train, batch_size=16, epochs=50, verbose=1)
	score = model.evaluate(X_test, y_test, batch_size=16)

	# Plot training & validation accuracy values
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('Model accuracy')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Epoch')
	# plt.legend(['Train', 'Test'], loc='upper left')
	# plt.show()

	# Plot training & validation loss values
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('Model loss')
	# plt.ylabel('Loss')
	# plt.xlabel('Epoch')
	# plt.legend(['Train', 'Test'], loc='upper left')
	# plt.show()

	model.save("model.h5")

	return model, score

model, score = train_model()
# model = load_model("model.h5")