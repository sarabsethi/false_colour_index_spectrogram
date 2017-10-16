import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.stats
import os
import glob

# Here we'll calculate entropy of each frequency bin separately
def fentropy(gram):
    return np.array([scipy.stats.entropy(row) for row in gram])

# Calculate ACI of spectrogram
def aci(gram):
    return np.array([np.sum(np.abs(row[1:] - row[:-1]))/np.sum(row) for row in gram])

if __name__ == '__main__':

    # Get local folder with separate wav files
    dir_path = os.path.dirname(os.path.realpath(__file__))
    audio_data_dir = os.path.join(dir_path, 'input_audio')
    audio_files = glob.glob('{}/*.wav'.format(audio_data_dir))
    audio_files = os.listdir(audio_data_dir)
    print('{} input audio files'.format(len(audio_files)))

    output_dir = './output_spectrograms'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    aci_list = []
    fentropy_list = []
    mag_list = []

    for idx, fname in enumerate(audio_files):
        full_audio_path = os.path.join(audio_data_dir,fname)

        # Calculate spectrogram and median filter
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

    # Visualise false colour spectrogram
    np.save(os.path.join(output_dir,'fentropy_spec.npy'),multifentropy)
    np.save(os.path.join(output_dir,'aci_spec.npy'),multiaci)
    np.save(os.path.join(output_dir,'mags.npy'),mags)

    # Scale the individual spectrograms to the range [0,1]
    multifentropy_scaled = (multifentropy - np.amin(multifentropy)) / np.amax(multifentropy)
    multiaci_scaled = (multiaci - np.amin(multiaci)) / np.amax(multiaci)
    mags_scaled = (mags - np.amin(mags)) / np.amax(mags)

    false_colour_image = [multifentropy_scaled, multiaci_scaled, mags_scaled]
    plt.imshow(np.transpose(false_colour_image, axes=(1,2,0)), aspect='auto', origin='lower')
    plt.xlabel('Which soundfile (~time)')
    plt.ylabel('Frequency')
    plt.show()
