import argparse
import os, glob
import numpy as np
from scipy.stats import entropy
import matplotlib

import soundfile as sf
import matplotlib.pyplot as plt
import librosa
from multiprocessing import Pool

profiling = False  # option to profile code using cPython

############################################################
# constants


indices_to_calc = ['fentropy', 'aci', 'specpow']  # these are the stats that we use for the RGB channels. Output saved as profilling.prof
chunk_length_seconds = 60  # in seconds
process_in_chunks = True

default_in  = './input_audio/SMM05736_20230920_180556.flac'
default_out = 'default_output'

perc_cutoff = 10

############################################################
# some acoustic indices

def fentropy(spectro):
	"Calculate entropy of each frequency bin separately"
	return np.array([entropy(row) for row in spectro])

def aci(spectro):
	"Calculate ACI of spectrogram for each freq bin separately"
	return np.array([np.sum(np.abs(row[1:] - row[:-1]))/np.sum(row) for row in spectro])

def magsum(spectro):
	"Calculate sum-of-magnitudes for each freq bin separately"
	return np.sum(spectro, axis=1)

def specpow(spectro):
	"Calculate power for each freq bin separately"
	return np.mean(spectro * spectro, axis=1)

############################################################
# functions to calc the stats

def get_supported_audio_files(directory):
    # Get a dictionary of supported audio formats
    formats = sf.available_formats()
    
    # We'll assume that the format codes from `available_formats()` correspond to common file extensions.
    # This might not always be the case, and you might need to manually map them to correct file extensions.
    supported_extensions = set(formats.keys())
    
    # Add common audio file extensions that are related to the supported formats
    supported_extensions.update(['wav', 'flac', 'aiff', 'ogg', 'mp3', 'aac'])

    # Prepare to collect supported audio files
    supported_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check file extension
            extension = file.split('.')[-1].lower()
            if extension in supported_extensions:
                # Add file to list if its extension is supported
                supported_files.append(os.path.join(root, file))

    return supported_files

def calculate_index_spectrogram_singlecolumn(spectro):
	"""Calculates a single 'column', i.e. it assumes the supplied spectrogram represents a single temporal window to be summarised into a single column pixel."""
	stats = [
		fentropy(spectro),
		aci(spectro),
		specpow(spectro),
	]
	return stats

def calculate_index_spectrograms(path, hop_percent=100):
		
	# get the current file's sample rate
	audiosr = librosa.get_samplerate(path)

	if process_in_chunks:
		chunk_length = int(chunk_length_seconds * audiosr)
	else:
		chunk_length = len(librosa.load(path, sr=None)[0])

	hop_length_samples = int(chunk_length * hop_percent / 100.)
	
	for block in sf.blocks(path, blocksize=chunk_length, overlap=(hop_length_samples - chunk_length)):
		spectro = abs(librosa.core.stft(block))  #[1x1025]
		somedata = calculate_index_spectrogram_singlecolumn(spectro) # [3x1025]
		yield somedata


def calculate_and_write_index_spectrograms(infpath, output_dir, hoppc=100):
	"""simply iterates over the 1ry func and writes out files in our standard format (each channel a separate file, and each file a npz array)"""

	# Make output directory if it doesn't exists
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# create an empty list containing a list for each in indices_to_calc
	arrays = [[] for statname in indices_to_calc]
 
	for results in calculate_index_spectrograms(infpath, hop_percent=hoppc):
		for (anarray, aresult) in zip(arrays, results):
			anarray.append(aresult)

	for (anarray, astatname) in zip(arrays, indices_to_calc):
		anarray = np.array(anarray)
		print(np.shape(anarray))
		np.savez(os.path.join(output_dir, 'indexdata_%s.npz' % astatname), specdata=anarray)

########################
def plot_fci_spectrogram(data_dir, doscaling=True, Fs=44100, hoppc=100, fmin=0, fmax=None):
	"Composes a false-colour spectrogram plot from precalculated data. Returns the Matplotlib figure object, so you can show() it or plot it out to a file."


	if fmin<0:
		raise ValueError("Negative minimum frequency was requested. %g" % fmin)
	if fmax is None or fmax>(Fs*0.5):
		fmax = Fs*0.5

	false_colour_image = np.array([np.load(os.path.join(data_dir, 'indexdata_%s.npz' % statname))['specdata'] for statname in indices_to_calc])
	false_colour_image = np.transpose(false_colour_image, axes=(2,1,0))

	print(np.shape(false_colour_image))
	print('max {} min {}'.format(np.max(false_colour_image, axis=(0,1)), np.min(false_colour_image, axis=(0,1))))

	if doscaling:
		#print(np.shape(false_colour_image))
		#print(np.shape(np.percentile(false_colour_image, perc_cutoff, axis=(0,1), keepdims=True)))
		false_colour_image = (false_colour_image - np.percentile(false_colour_image, perc_cutoff, axis=(0,1), keepdims=True)) \
		                                         / np.percentile(false_colour_image, 100-perc_cutoff, axis=(0,1), keepdims=True)

	print('max {} min {}'.format(np.max(false_colour_image, axis=(0,1)), np.min(false_colour_image, axis=(0,1))))

	maxtime = np.shape(false_colour_image)[1] * (float(chunk_length_seconds)/60.) * (hoppc/100.)  # NB assumes chunking was performed using chunks of size "chunk_length", which is not always true
	#print(maxtime)
	timeunits = 'minutes'
	if maxtime > 180:
		maxtime /= 60.
		timeunits = "hours"

	matplotlib.rcParams.update({'font.size': 12})
	matplotlib.rcParams.update({'font.family' : 'serif'})
	fig=plt.figure(figsize=(15, 5), facecolor='white')
	plt.imshow(false_colour_image, aspect='auto', origin='lower', interpolation='none', extent=(0, maxtime, 0, Fs/2000))
	plt.ylim(fmin/1000, fmax/1000)
	plt.xlabel('Time (%s)' % timeunits)
	plt.ylabel('Frequency (kHz)')
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)
	plt.tight_layout()
	return plt.gcf()

########################

def main():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", default=default_in, help="Input path: can be a path to a single file (which will be chunked), or a folder full of wavs, or the input can be a list of wav files which you explicitly specify")
	parser.add_argument("-o", default=default_out, type=str, help="Output path: a folder (which should exist already) in which data files will be written.")
	#parser.add_argument("-c", default=1, type=int, choices=[0,1], help="Whether to calculate the stats afresh. Use -c=0 to reuse previously calculated stats.")
	parser.add_argument("-n", default=1, type=int, choices=[0,1], help="Whether to apply scaling (normalisation) of the statistics before plotting them.")
	parser.add_argument("--savef", default='', type=str, help="Image file where the output figure should be saved (including extension (png, jpg, etc.). Expect issues with vector graphics")
	parser.add_argument("--hop", default=100, type=float, help="How much to advance each 'frame', as a percentage of the 1-min framesize. Default of 100%% is recommended for long >2hr. For 1hr audio you could try 25 for finer resolution.")
	parser.add_argument("--fmin", default=0, type=float, help="Lowest frequency (in Hz) to show on the plot.")
	parser.add_argument("--fmax", default=44100, type=float, help="Highest frequency (in Hz) to show on the plot.")
	args = parser.parse_args()
	
	input_path = args.i
	print(input_path)
 
	# decide if the in_path is a file or a directory
	if os.path.isfile(input_path):
		
		# make a singleton list so the loop still works
		path_list = [input_path]
		print(f'processing file {input_path} ')

	elif os.path.isdir(input_path):
     
		# we have a dir so list the audio files
		path_list = get_supported_audio_files(input_path)
		print(f'processing folder {input_path} ')
	else:
		raise ValueError("Path not found: %s" % path_list)
  
	for path in path_list:
		print(f'processing file {path}')
		if profiling:
			# if the code is being profilled import cPython
			import cProfile
			import pstats

			pr =  cProfile.Profile()
			pr.enable()
			calculate_and_write_index_spectrograms(infpath=args.inpaths, output_dir=args.o, hoppc=args.hop)
			pr.disable()
			stats = pstats.Stats(pr)
			stats.sort_stats(pstats.SortKey.TIME)
			# Now you have two options, either print the data or save it as a file
			stats.print_stats() # Print The Stats
			stats.dump_stats("profilling.prof") # Saves the data in a file, can me used to see the data visually
		else:
			calculate_and_write_index_spectrograms(path, output_dir=args.o, hoppc=args.hop)
   
      
	# now plot
	ourplot = plot_fci_spectrogram(args.o, doscaling=args.n, hoppc=args.hop, fmin=args.fmin, fmax=args.fmax)
	if not args.savef == "":
		ourplot.savefig(args.savef)
	else:
		ourplot.show()
		input("Press a key to close")
    
########################
if __name__=='__main__':
    
    main()
