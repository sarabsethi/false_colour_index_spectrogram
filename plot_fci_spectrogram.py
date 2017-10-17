import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import glob

# Load calculated index spectrograms from files
output_dir = './output_spectrograms'
multiaci = np.load(os.path.join(output_dir,'aci_spec.npy'))
multifentropy = np.load(os.path.join(output_dir,'fentropy_spec.npy'))
mags = np.load(os.path.join(output_dir,'mags.npy'))

# Scale the individual spectrograms to the range [0,1], in a robust(ish) way to outliers
# NOTE: Towsey et. al. don't recommend this scaling approach as it doesn't allow
# comparisons across different datasets, but for one-off visualisations it works well
perc_cutoff = 10
multifentropy_scaled = (multifentropy - np.percentile(multifentropy,perc_cutoff)) / np.percentile(multifentropy,100-perc_cutoff)
multiaci_scaled = (multiaci - np.percentile(multiaci,perc_cutoff)) / np.percentile(multiaci,100-perc_cutoff)
mags_scaled = (mags - np.percentile(mags,perc_cutoff)) / np.percentile(mags,100-perc_cutoff)

# Show false colour index spectrogram (assume 44.1kHz sampling rate)
false_colour_image = [multifentropy_scaled, multiaci_scaled, mags_scaled]
Fs = 44100
hours = multifentropy.shape[1] / 60
plt.imshow(np.transpose(false_colour_image, axes=(1,2,0)), aspect='auto', origin='lower', interpolation='none', extent=(0,hours,0,Fs/2))
plt.xlabel('Time (+ hours)')
plt.ylabel('Frequency (Hz)')
plt.show()
