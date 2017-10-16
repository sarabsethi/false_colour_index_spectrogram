import os
import subprocess

# Get local folder to save audio in
dir_path = os.path.dirname(os.path.realpath(__file__))
audio_data_dir = os.path.join(dir_path,'aac_audio')
full_aac_files = os.listdir(audio_data_dir)
full_aac_files = sorted(full_aac_files)

for full_aac in full_aac_files:
    if not full_aac.endswith('secs.aac'):
        continue

    # split 20 minute file into 1 minute segments
    full_aac_path = os.path.join(audio_data_dir,full_aac)
    print('Splitting full AAC file into 1 minute segments')
    subprocess.call('ffmpeg -i {} -f segment -segment_time 60 -c copy {}_%02d.aac'.format(full_aac_path,full_aac_path),shell=True)

    print('Removing local copy of full length file {}'.format(full_aac))
    os.remove(full_aac_path)

# Now transcode the 1 minute aac files to wav format
short_aac_files = os.listdir(audio_data_dir)
short_aac_files = sorted(short_aac_files)

for short_aac in short_aac_files:
    if not short_aac.endswith('.aac'):
        continue

    short_aac_path = os.path.join(audio_data_dir,short_aac)
    short_wav_path_stereo = short_aac_path + '_stereo.wav'
    short_wav_path = short_aac_path + '.wav'

    # transcode aac file to wav
    print('Transcoding file {} to wav'.format(short_aac))
    subprocess.call('faad {} -o {}'.format(short_aac_path,short_wav_path_stereo),shell=True)
    print('Removing short aac file {}'.format(short_aac))
    os.remove(short_aac_path)

    # faad annoyingly makes everything stereo so mix back down to mono
    subprocess.call('ffmpeg -i {} -ac 1 {}'.format(short_wav_path_stereo,short_wav_path),shell=True)
    print('Removing short stereo wav file {}'.format(short_aac))
    os.remove(short_wav_path_stereo)
