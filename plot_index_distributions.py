import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import glob

output_dir = './output_spectrograms'

aci_spec = np.load(os.path.join(output_dir,'aci_spec.npy'))
fentropy_spec = np.load(os.path.join(output_dir,'fentropy_spec.npy'))
mag_spec = np.load(os.path.join(output_dir,'mags.npy'))

plt.subplot(311)
plt.hist(aci_spec.flatten())
plt.xlabel('ACI')
plt.ylabel('Frequency')

plt.subplot(312)
plt.hist(fentropy_spec.flatten())
plt.xlabel('Frequency Entropy')
plt.ylabel('Frequency')

plt.subplot(313)
plt.hist(mag_spec.flatten())
plt.xlabel('Magnitude')
plt.ylabel('Frequency')

plt.show()
