# https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network/blob/master/eeg_lstm-v2.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

log = open("eeg_test.log", "w")

eeg_df = pd.read_csv("eeg_test.csv", sep="\t")
eeg_df = eeg_df[["Time", "thumb_near"]]	# drop all columns except Time and thumb_near
log.write(eeg_df.describe().to_string())

ax = eeg_df.plot(x="Time", y="thumb_near")
plt.show()

# Keras LSTM (long short-term memory)
from sklearn import preprocessing


# parameters from gdf file
n_features = 96		# number of channels
time_steps = 321998	# number of ms each sample is run for
event_types = 10	# event types are [768, 785, 786, 1536, 1537, 1538, 1539, 1540, 1541, 1542]

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
