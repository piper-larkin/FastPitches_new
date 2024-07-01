import librosa
'''
Edited from local version
'''

# Path to just training files
file_list_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/80s_reagan_1/reagan_audio_pitch_1980s_text_dev_2.txt'
total_duration = 0.0
file_paths = []
# line is like: mels/reagan_1981_18_11_049.pt|pitch/reagan_1981_18_11_049.pt|They can preser..
with open(file_list_path, 'r') as file:
    for line in file:
        # line like: mels/LJ050-0234.pt|pitch/LJ050-0234.pt|It has used other ...
        columns = line.strip().split('|')
        file_name = columns[0]
        label = file_name[5:-3]
        # file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/' + label + '.wav'
        file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/80s_reagan_2/wavs/' + label + '.wav'
        file_paths.append(file_path)

for wav_file in file_paths:
    y, sr = librosa.load(wav_file)
    total_duration += librosa.get_duration(y, sr)
print('Seconds:', total_duration)
print('Minutes:', total_duration / 60)
print('Hours:', total_duration / 60 /60)