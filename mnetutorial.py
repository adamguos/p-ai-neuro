# https://mne.tools/stable/auto_tutorials/epochs/plot_epoching_and_averaging.html#sphx-glr-auto-tutorials-epochs-plot-epoching-and-averaging-py

import os.path as op
import numpy as np

import mne

# Import MNE sample data
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
raw = mne.io.read_raw_fif(fname)
raw.set_eeg_reference("average", projection=True)	# set EEG average reference

# Plot raw data
order = np.arange(raw.info["nchan"])
order[9] = 312	# Exchange the plotting order of two channels
order[312] = 9	# to show the trigger channel as the 10th channel
# raw.plot(n_channels=10, order=order, block=True)

# Find events to split the continuous time series into epochs for
events = mne.find_events(raw)
print("Found %s events, first five:" % len(events))
print(events[:5])