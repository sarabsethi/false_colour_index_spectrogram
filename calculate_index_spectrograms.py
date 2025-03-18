import argparse, os
import numpy as np, matplotlib, matplotlib.pyplot as plt
from scipy.stats import entropy
from datetime import datetime
import librosa, soundfile as sf

# multiprocessing libraries
from multiprocessing import Manager
import multiprocessing, psutil
from queue import Empty

############################################################
# constants

# define which stats to run, must match function names exactly
indices_to_calc = ['fentropy', 'aci', 'magsum']  # in order of [Red Green Blue] channels.
chunk_length_seconds = 60  # in seconds
process_in_chunks = True
doscaling = True 

default_in  = './input_audio'
default_out = 'default_output'

# which percentile to cut off from the plots
perc_cutoff = 10

# dont use logical cores as hyper threading is slower
physical_cores = psutil.cpu_count(logical = False)

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

	if len(indices_to_calc) == 3:
		stats = []
		for acoustic_index in indices_to_calc:
			# Check if the function exists
			if acoustic_index in globals() and callable(globals()[acoustic_index]):
				# Call the function and append the result to the output list
				stats.append(globals()[acoustic_index](spectro))
			else:
				raise Exception(f"Function '{acoustic_index}' not found, acoustic indices must match valid functions")
	else:
		raise Exception("There must be 3 acoustic indices defined in indices_to_calc")

	return stats


 		#stats = [
		#aci(spectro), 		# red
  		#fentropy(spectro), 	# green
  		#specpow(spectro), 	# blue

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

def calculate_index_spectrograms(path, hop_percent=100):
    # reads an audio file in blocks and yields blocks as iterable.
	# each block yields a 3x1025 list for 3 acoustic indices and 1025 stft bins
    
	# TODO Change hop length to percent overlap
	with sf.SoundFile(path, 'r') as file:
     
		# calculate 60 seconds in samples
		if process_in_chunks:
			chunk_length = int(chunk_length_seconds * file.samplerate)
		else:
			chunk_length = file.frames
   		
		hop_length_samples = int(chunk_length * hop_percent / 100.)

		#print(f"{datetime.now()}: generating stats for {path}")
		for block in file.blocks(chunk_length, hop_length_samples - chunk_length):
			spectro = abs(librosa.core.stft(block))  #[1x1025]		
			stats = calculate_index_spectrogram_singlecolumn(spectro)
			yield stats
		#print(f"{datetime.now()}: completed calculating stats for {path}")
 
def process_file(path, hop_percent, shared_dict, worker_name):
	"""
	function sent to a worker for each file. Results returned to shared_dict
	"""
	# create an empty list containing a list for each in indices_to_calc
	result_arrays = [[] for statname in indices_to_calc]

	# iterate through 60 second blocks of audio
	for block in calculate_index_spectrograms(path, hop_percent=hop_percent):
		for (results, result) in zip(result_arrays, block):
			results.append(result)
	
 	# tell the user the worker completed the task
	print(f'{datetime.now()}: [{worker_name}] complete {path} ')
 
	# add the result to the shared dict
	shared_dict.update({path:result_arrays})
	

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
def worker_process(queue):
	while True:
		try:	
			args = queue.get(timeout=1)
			if args is None:
				break
			print(f"{datetime.now()}: [{multiprocessing.current_process().name}] starting: {args[0]}")
			process_file(*args, multiprocessing.current_process().name)
		except Exception:
			break

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
	output_dir=args.o
	print(input_path)
 
	# decide if the in_path is a file or a directory
	if os.path.isfile(input_path):
		
		# make a singleton list so the loop still works
		paths = [input_path]
		print(f'{datetime.now()}: processing file {input_path} ')

	elif os.path.isdir(input_path):
		# we have a dir so make a list of supported audio files
		paths = get_supported_audio_files(input_path)
		
		# sort the paths by string
		paths = sorted(paths)
		print(f'{datetime.now()}: processing folder {input_path} ')
	else:
		raise ValueError("Path not found: %s" % paths)
  
	# if the output folder doesn't exist, create it
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# set the number of workers based on physical cores and number of tasks
	if len(paths) >= physical_cores:
		num_workers = physical_cores
	else:
		num_workers = len(paths)
		
     
    # create a multiprocessing manager object
	with Manager() as manager:

		# create shared variable for results and tasks
		shared_dict = manager.dict()
	
		# create a queue
		queue = manager.Queue()
	
		# add audio files to process to queeue
		for path in paths:
			queue.put((path, 100, shared_dict))

		processes = [multiprocessing.Process(target=worker_process, args=(queue,), name=f"Worker-{i+1}") for i in range(num_workers)]
		
		for p in processes:
			p.start()

		# Wait for all processes to finish
		for p in processes:
			p.join()

		results = [np.array(shared_dict[path]) for path in paths]
	# The output is in results shape (n_tasks,3,60,1025)
	# print(results)
	
	print(f"{datetime.now()}: all processes have finished.")
	results = np.concatenate(results, axis=1)
 
	for stat in range(results.shape[0]):
		specdata = results[stat]
		np.savez(os.path.join(output_dir, f'indexdata_{indices_to_calc[stat]}.npz'), specdata=specdata)
		np.savetxt(os.path.join(output_dir, f'indexdata_{indices_to_calc[stat]}.csv'), specdata, delimiter=",")
  
  
	# now plot
	ourplot = plot_fci_spectrogram(args.o, doscaling=args.n, hoppc=args.hop, fmin=args.fmin, fmax=args.fmax)
	
	if not args.savef == "":
		ourplot.savefig(args.savef)
	else:
		current_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
		ourplot.savefig(f'{current_timestamp}_false_colour_plot.png')
		ourplot.show()
		input("Press a key to close")

########################
if __name__=='__main__':
    main()