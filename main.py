import os
import subprocess
from compute_indice import *
from acoustic_index import *
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib.colors import LogNorm

# Set config file
yml_file = 'acoustic_indices_config.yaml'
with open(yml_file, 'r') as stream:
    data_config = yaml.load(stream)
ci = data_config['Indices']

# Get local folder to save audio in
dir_path = os.path.dirname(os.path.realpath(__file__))
audio_data_dir = os.path.join(dir_path, 'input_audio')

audio_files = os.listdir(audio_data_dir)
audio_files = sorted(audio_files)

aci_vals = []
h_t_vals = []
power_specs = []

for audio_file in audio_files:
    if not audio_file.endswith('.wav'):
        continue

    audio_file_path = os.path.join(audio_data_dir, audio_file)

    print('Calculating indices for {}'.format(audio_file))

    file = AudioFile(audio_file_path, verbose=True)

    # Acoustic complexity index
    index_name = 'Acoustic_Complexity_Index'
    print('\tCompute', index_name)
    spectro, _ = compute_spectrogram(file, **ci[index_name]['spectro'])
    j_bin = ci[index_name]['arguments']['j_bin'] * file.sr / ci[index_name]['spectro']['windowHop'] # transform j_bin in samples
    aci_vals.append(compute_ACI(spectro, j_bin)[0])

    # H[t]
    index_name = 'Temporal_Entropy'
    print('\tCompute', index_name)
    # Note H_t is reversed as in Towsey et. al. 2014
    h_t_vals.append(1 - compute_TH(file))

    # Power spectrum (aka one column of the spectrogram)
    print('\tComputing fft (one column of spectrogram)')
    power_specs.append(spectro)

    # print('Removing audio file {}'.format(audio_file))
    # os.remove(audio_file_path)

#print(aci_vals)
#print(h_t_vals)
#print(power_specs)

spectrogram = np.asarray(power_specs)
spectrogram = np.transpose(spectrogram)
spectrogram = spectrogram[0:spectrogram.shape[0]//2,:]

ACI_spec = aci_vals * spectrogram
ACI_spec = (ACI_spec - np.amin(ACI_spec)) / np.amax(ACI_spec)

dummy_spec = np.zeros(spectrogram.shape)

H_t_spec = h_t_vals * spectrogram
H_t_spec = (H_t_spec - np.amin(H_t_spec)) / np.amax(H_t_spec)

rgb_spectrogram = np.zeros((spectrogram.shape[0],spectrogram.shape[1],3), 'uint8')
rgb_spectrogram[..., 0] = ACI_spec*256
rgb_spectrogram[..., 1] = H_t_spec*256
rgb_spectrogram[..., 2] = dummy_spec*256

plt.imshow(rgb_spectrogram, origin='lower', aspect=spectrogram.shape[1]/spectrogram.shape[0], interpolation='none')
plt.show()
