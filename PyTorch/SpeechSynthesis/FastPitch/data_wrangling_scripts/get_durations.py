import librosa
from collections import defaultdict 
'''
Edited from local version
'''

## BY SPEAKER
# without black
speakers = ['bowman', 'byrne', 'cooke', 'cronkite', 'doyle', 'dunne', 'finucane', 'gogan', 'lawlor', 'lockwood', 'magee', 'nibhriain', 'odulaing', 'plomley', 'queen', 'thatcher', 'reagan']

speaker_times = defaultdict(tuple)
for speaker in speakers: 
    file_list_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_text_spk_age.txt'
    total_duration = 0.0
    file_paths = []
    # line is like: mels/reagan_1981_18_11_049.pt|pitch/reagan_1981_18_11_049.pt|They can preser..
    with open(file_list_path, 'r') as file:
        for line in file:
            # line like: mels/LJ050-0234.pt|pitch/LJ050-0234.pt|It has used other ...
            columns = line.strip().split('|')
            file_name = columns[0]
            label = file_name[5:-3]

            if speaker in label:
                # file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/' + label + '.wav'
                # file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/' + label + '.wav'
                file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/' + label + 'wav'
                file_paths.append(file_path)

    for wav_file in file_paths:
        y, sr = librosa.load(wav_file)
        total_duration += librosa.get_duration(y, sr)

    speaker_times[speaker] = (total_duration, (total_duration / 60), (total_duration / 60 /60))
    # print('Seconds:', total_duration)
    # print('Minutes:', total_duration / 60)
    # print('Hours:', total_duration / 60 /60)

print(speaker_times)


#### FOR WHOLE FILE:

# # Path to just training files
# file_list_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_text_spk_age.txt'
# total_duration = 0.0
# file_paths = []
# # line is like: mels/reagan_1981_18_11_049.pt|pitch/reagan_1981_18_11_049.pt|They can preser..
# with open(file_list_path, 'r') as file:
#     for line in file:
#         # line like: mels/LJ050-0234.pt|pitch/LJ050-0234.pt|It has used other ...
#         columns = line.strip().split('|')
#         file_name = columns[0]
#         label = file_name[5:-3]

#         if 'cooke' in label:
#             # file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/' + label + '.wav'
#             # file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/' + label + '.wav'
#             file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/' + label + 'wav'
#             file_paths.append(file_path)

# for wav_file in file_paths:
#     y, sr = librosa.load(wav_file)
#     total_duration += librosa.get_duration(y, sr)
# print('Seconds:', total_duration)
# print('Minutes:', total_duration / 60)
# print('Hours:', total_duration / 60 /60)