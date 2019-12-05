# https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network/blob/master/eeg_lstm-v2.ipynb

import matplotlib.pyplot as plt
import pandas as pd

eeg_df = pd.read_csv("eeg_test.csv", sep="\t")
# eeg_df = eeg_df[["Time", "thumb_near"]]	# drop all columns except Time and thumb_near
eeg_df = eeg_df.dropna()	# drop all rows without a thumb_near value
# eeg_df = (eeg_df - eeg_df.min()) / (eeg_df.max() - eeg_df.min())	# normalise all columns to be between 0 and 1

events_df = pd.read_csv("eeg_events.csv", sep="\t")
events_df = events_df[(events_df.type >= 1536) & (events_df.type <= 1542)]	# only keep rows corresponding to hand movement events

num_of_samples = events_df.shape[0]	# num of samples (hand movement events), given by num of rows
n_features = eeg_df.shape[1] - 1	# num of eeg channels, given by num of columns subtract time column

def slice_eeg_into_samples(eeg, events):
	samples_list = []

	for index, row in events.iterrows():
		index_range = eeg[eeg["Time"] >= row["latency"]].index[0]	# index of first eeg datapoint that happens after this event
		print("index range:", index_range)
		this_slice = eeg[:index_range + 1]	# take slice of datapoints that happened before this event
		
		samples_list.append(this_slice)
		eeg = eeg[index_range:].reset_index(drop=True)	# drop rows that happened before this event
	
	# the loop leaves out measurements of the last event, since each iteration appends datapoints that occur before this event
	# add it here
	samples_list.append(eeg)

	# the loop includes the first measurements, before any event takes place
	# drop it here
	samples_list.pop(0)
	
	return samples_list

out = slice_eeg_into_samples(eeg_df, events_df)