# https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network/blob/master/eeg_lstm-v2.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical

eeg_df = pd.read_csv("eeg_test.csv", sep="\t")
# eeg_df = eeg_df[["Time", "thumb_near"]]	# drop all columns except Time and thumb_near
eeg_df = eeg_df.dropna()	# drop all rows without a thumb_near value
# eeg_df = (eeg_df - eeg_df.min()) / (eeg_df.max() - eeg_df.min())	# normalise all columns to be between 0 and 1

events_df = pd.read_csv("eeg_events.csv", sep="\t")
events_df = events_df[(events_df.type >= 1536) & (events_df.type <= 1542)]	# only keep rows corresponding to hand movement events

num_of_samples = events_df.shape[0]	# num of samples (hand movement events), given by num of rows
n_features = eeg_df.shape[1] - 1	# num of eeg channels, given by num of columns subtract time column
time_steps = 2096

def slice_eeg_into_samples(eeg, events):
	samples_list = []

	for index, row in events.iterrows():
		index_range = eeg[eeg["Time"] >= row["latency"]].index[0]	# index of first eeg datapoint that happens after this event
		this_slice = eeg[:index_range + 1]	# take slice of datapoints that happened before this event
		this_slice = this_slice.drop("Time", axis=1)	# drop time column
		
		samples_list.append(this_slice)
		eeg = eeg[index_range:].reset_index(drop=True)	# drop rows that happened before this event
	
	# samples are not exactly the same length
	# pad on filler entries to make every sample as long as the longest one
	samples_list = pad_samples(samples_list)
	
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

def pad_samples(eeg_old):
	eeg_new = []

	longest = 0
	for sample in eeg_old:
		l = len(sample.index)
		if l > longest:
			longest = l
	
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

eeg_samples = slice_eeg_into_samples(eeg_df, events_df)
events_1hot, lb = one_hot_events(events_df)

# event_types = events_df["type"].to_numpy()
# event_types = event_types - event_types.min() + 1
# events_1hot = to_categorical(event_types)

X_train, X_test, y_train, y_test = split_train_test(eeg_samples, events_1hot)

# Train LSTM model, using options from EEG paper linked at the top
# dense layer for 1_hot: https://stackoverflow.com/questions/49604765/create-model-using-one-hot-encoding-in-keras
# model1 = Sequential()
# model1.add(Dense(512, input_shape=(time_steps, n_features)))
# model1.add(LSTM(128))
# model1.add(Dense(7, activation="softmax"))

# model1.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# model1.fit(X_train, y_train, batch_size=16, epochs=42)
# score = model1.evaluate(X_test, y_test, batch_size=16)

# model1.save("model1.h5")

model2 = Sequential()
model2.add(LSTM(100, return_sequences=False, input_shape=(time_steps, n_features)))
model2.add(Dropout(0.5))
model2.add(Dense(7, activation="sigmoid"))

model2.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model2.fit(X_train, y_train, batch_size=1, epochs=42)
score = model2.evaluate(X_test, y_test, batch_size=1)

model2.save("model2.h5")