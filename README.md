# false_colour_index_spectrogram
Generate a false colour index spectrogram to nicely visualise long duration soundscape recordings - based on [Towsey et. al. 2014](http://www.sciencedirect.com/science/article/pii/S1877050914002403)

## Scripts
`calculate_index_spectrograms.py` - if you run this it calculates the indices for the false colour index spectrogram, and also plot the resulting false-colour spectrogram. This file also contains the main functions which can be used in your own Python scripts if you import the file as a module. When using the file as a script, you can optionally change the input/output paths and you can also disable re-calculation of the statistics: run `python calculate_index_spectrograms.py -h` for information about the command-line options.

`plot_index_distributions.py` - used to look at the distribution of values for each of the three indices

### Example output plot:
24 hours recorded from a tropical rainforest in Sabah, Borneo. Dawn and dusk choruses are clearly visible with different patterns of calling during the day and night time
![Example 24 hour false colour index spectrogram](https://raw.githubusercontent.com/sarabsethi/false_colour_index_spectrogram/master/example_24_hrs.JPG)

## Input / output data format
Input can be a single long file which will be analysed in 1-minute chunks, OR a folder containing a series of WAV files (we assume they are 1 minute long).

Long-duration recordings in single large WAV files can be handled by this script. On our laptop it takes about 30min to analyse 12 hours (i.e. 24x speed) -- the time taken will scale linearly with the length of the input. Note that it's NOT any faster to split the audio into chunks first.

Numpy ndarrays storing individual index spectrograms are stored in ./output_spectrograms/ folder.

## Authors
* [Sarab Sethi](http://www.imperial.ac.uk/people/s.sethi16) (Imperial College London)
* [Dan Stowell](http://mcld.co.uk/research/) (Queen Mary University of London)

