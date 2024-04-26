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

default_in  = 'SMM05736_20230920_180556.flac'
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

def calculate_index_spectrogram_singlecolumn(spectro):
	"""Calculates a single 'column', i.e. it assumes the supplied spectrogram represents a single temporal window to be summarised into a single column pixel."""
	stats = [
		fentropy(spectro),
		aci(spectro),
		specpow(spectro),
	]
	return stats

def calculate_index_spectrograms(path_list, hop_percent=100):
    # takes a folder or file name
    # returns 
	"""Supply either a list of wav file paths, or a folder to be globbed, or a single file path (to be chopped into 60s chunks).
	This is a generator function which will create an ITERATOR, returning one new column of index-pixel data on every loop of the iter."""
	
	if type(path_list) is str:
		if not os.path.exists(path_list):
			raise ValueError("Path not found: %s" % path_list)
	else:
		for path in path_list:
			if not os.path.exists(path):
				raise ValueError("Path not found: %s" % path)

	#############################################
	# Handle user argument - either a foldername, or a list of files, or a single file which we will auto-chop
 
	if type(path_list) is not str:
		# a list has been supplied. could be from the CLI arguments, even if it's a single item.
		if len(path_list)==1:  # in this special case, we strip off the list so the next check can determine if it's a folder to glob
			path_list = path_list[0]
   
	if type(path_list) is str:
		if os.path.isdir(path_list):
			# if a single folder, we auto-expand it to a sorted list of the files found immediately within that folder
			path_list = sorted(glob.iglob(os.path.join(path_list, "*.wav")))
		else:
			# a single non-directory item submitted? then convert to a singleton list, and YES we'll chop it
			path_list = [path_list]
			process_in_chunks = True

	#############################################
	# at this point we expect infpath to be a list of wav filepaths. (if the user submitted a list of something-elses, this assumption could break. caveat emptor)
	path_list = sorted(path_list)
 
	for path in path_list:
		
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
	parser.add_argument("inpaths", nargs='*', default=default_in, help="Input path: can be a path to a single file (which will be chunked), or a folder full of wavs, or the input can be a list of wav files which you explicitly specify")
	parser.add_argument("-o", default=default_out, type=str, help="Output path: a folder (which should exist already) in which data files will be written.")
	#parser.add_argument("-c", default=1, type=int, choices=[0,1], help="Whether to calculate the stats afresh. Use -c=0 to reuse previously calculated stats.")
	parser.add_argument("-n", default=1, type=int, choices=[0,1], help="Whether to apply scaling (normalisation) of the statistics before plotting them.")
	parser.add_argument("--savef", default='', type=str, help="Image file where the output figure should be saved (including extension (png, jpg, etc.). Expect issues with vector graphics")
	parser.add_argument("--hop", default=100, type=float, help="How much to advance each 'frame', as a percentage of the 1-min framesize. Default of 100%% is recommended for long >2hr. For 1hr audio you could try 25 for finer resolution.")
	parser.add_argument("--fmin", default=0, type=float, help="Lowest frequency (in Hz) to show on the plot.")
	parser.add_argument("--fmax", default=44100, type=float, help="Highest frequency (in Hz) to show on the plot.")
	args = parser.parse_args()

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
		calculate_and_write_index_spectrograms(infpath=args.inpaths, output_dir=args.o, hoppc=args.hop)
      
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
