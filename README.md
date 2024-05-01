# false_colour_index_spectrogram
Generate a false colour index spectrogram to nicely visualise long duration soundscape recordings - based on [Towsey et. al. 2014](http://www.sciencedirect.com/science/article/pii/S1877050914002403)

## Scripts
`calculate_index_spectrograms.py` - 

-   Calculates acoutic indices and plots a false colour index spectrogram for audio files located in `./input_audio` folder.
-   Plots the resulting false-colour spectrogram as continuous plot
-   Assumes zero gap between input audio files.
  
Command line options:
* -i, default="./input_audio", "Input path: can be a path to a single file (which will be chunked), or a folder full of wavs, or the input can be a list of wav files which you explicitly specify"
* -o, default="./default_out", "Output path: a folder (which should exist already) in which data files will be written."
* -n, default=1, choices=[0,1], "Whether to apply scaling (normalisation) of the statistics before plotting them."
* --savef, default='YYYY-MM-DD_HHMM_false_colour_plot.png', "Image file where the output figure should be saved (including extension (png, jpg, etc.). Expect issues with vector graphics")
* --hop, default=100, "How much to advance each 'frame', as a percentage of the 1-min framesize. Default of 100%% is recommended for long >2hr. For 1hr audio you could try 25 for finer resolution."
* --fmin, default=0, "Lowest frequency (in Hz) to show on the plot."
* --fmax, default=44100, "Highest frequency (in Hz) to show on the plot."
   
This file also contains the main functions which can be used in your own Python scripts if you import the file as a module. When using the file as a script, you can optionally change the input/output paths and you can also disable re-calculation of the statistics: run `python calculate_index_spectrograms.py -h` for information about the command-line options.

`plot_index_distributions.py` - used to look at the distribution of values for each of the three indices

Code tested using Python 3.12 on an 6 core Intel i7 Macbook Pro at 2.6 GHz.

### Example output plot:
24 hours recorded from a tropical rainforest in Sabah, Borneo. Dawn and dusk choruses are clearly visible with different patterns of calling during the day and night time
![Example 24 hour false colour index spectrogram](https://raw.githubusercontent.com/sarabsethi/false_colour_index_spectrogram/master/example_24_hrs.JPG)

## Input / output data format
Input can be a single long file which will be analysed in 1-minute chunks, OR a folder containing a series of WAV files (we assume they are 1 minute long).

Long-duration recordings split into multiple WAV files of any length can be handled by this script. On a 6 core Intel i7 running MacOS 14.4.1 24 hours of audio can be processed in 120 seconds.

Numpy ndarrays storing individual index spectrograms are stored in ./output_spectrograms/ folder.

# change log
-   April 2024 (Joshua C Taylor)
    - replaced modded librosa module with soundfile
    - added multiprocessing ability
    - run time for 24 hours reduced from 1 hour to 120 seconds
    - edit for readability
   
## Authors
* [Joshua Taylor (updated to python 3.10 and added parallel processing)](https://socialenvironment.org.uk/about/) (Social Environment CIC) 
* [Sarab Sethi](http://www.imperial.ac.uk/people/s.sethi16) (Imperial College London)
* [Dan Stowell](http://mcld.co.uk/research/) (Queen Mary University of London)
