'''
Generate a false colour index spectrogram - based on Towsey et. al. - http://dx.doi.org/10.1016/j.procs.2014.05.063

"main_calc_fci_spectrogram.py" calculates the indices for the false colour index spectrogram
"plot_index_distributions.py" used to look at the distribution of values for each of the three indices
"plot_fci_spectrogram" to show false colour index spectrogram (without recalculating indices)

N.B. the script assumes series of 1 minute files are stored in ./input_audio/ dir

If you have long duration recordings, to split into individual files use ffmpeg:
"ffmpeg -i long_input_file.wav -f segment -segment_time 60 -c copy short_output_file_%02d.wav"
then transfer these one minute files to ./input_audio/ and you're ready to go

Numpy ndarrays storing individual index spectrograms are stored in ./output_spectrograms/ dir

Authors: Sarab Sethi (Imperial College London), Dan Stowell (Queen Mary University of London)
'''

import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import scipy.stats
import glob

# Calculate entropy of each frequency bin separately
def fentropy(gram):
    return np.array([scipy.stats.entropy(row) for row in gram])

# Calculate ACI of spectrogram
def aci(gram):
    return np.array([np.sum(np.abs(row[1:] - row[:-1]))/np.sum(row) for row in gram])

if __name__ == '__main__':

    # Get local folder with separate 1 minute wav files (see split_long_files.py)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    audio_data_dir = os.path.join(dir_path, 'input_audio')
    audio_files = glob.glob('{}/*.wav'.format(audio_data_dir))
    audio_files = os.listdir(audio_data_dir)
    print('{} input audio files'.format(len(audio_files)))

    # Make output directory
    output_dir = './output_spectrograms'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    aci_list = []
    fentropy_list = []
    mag_list = []

    for idx, fname in enumerate(audio_files):
        full_audio_path = os.path.join(audio_data_dir,fname)

        # Calculate spectrogram and median filter it
        print('Calculate spectrogram of {} ({}/{})'.format(fname,idx+1,len(audio_files)))
        spectro = abs(librosa.core.stft(librosa.load(full_audio_path)[0]))
        print('\tMedian filter spectrogram')
        spectro = np.maximum(spectro - np.median(spectro, axis=1, keepdims=True), 1e-3)

        # Calculate indices for specific audio file
        print('\tCalculate fentropy')
        fentropy_list.append(fentropy(spectro))
        print('\tCalculate ACI')
        aci_list.append(aci(spectro))
        print('\tCalculate magnitude')
        mag_list.append(np.sum(spectro, axis=1))

    # Concatenate lists into full spectrograms for each index
    multifentropy = np.asarray(fentropy_list).T
    multiaci = np.asarray(aci_list).T
    mags = np.asarray(mag_list).T

    # Save individual index spectrograms to file
    np.save(os.path.join(output_dir,'fentropy_spec.npy'),multifentropy)
    np.save(os.path.join(output_dir,'aci_spec.npy'),multiaci)
    np.save(os.path.join(output_dir,'mags.npy'),mags)

    # Run script to plot false colour index spectrogram (from saved files)
    os.system('python plot_fci_spectrogram.py')
