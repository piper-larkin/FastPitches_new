import os
import wave
import contextlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def get_wav_duration(wav_file):
    """Return the duration of a WAV file in seconds."""
    with contextlib.closing(wave.open(wav_file, 'r')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration

all_ages = {19, 21, 22, 24, 26, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 95}
audio_text_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_text_spk_age.txt'

# files_by_age = defaultdict(list)
# lengths_by_age = defaultdict(float)

# with open(audio_text_file, 'r') as file:
#     for line in file:
#         # like: wavs/queen_2008_3_1_007.wav|The impact of pollution falls unequally.|17|82
#         info = line.split('|')
#         age = int(info[-1])
#         file_name = info[0]

#         files_by_age[age].append(file_name)

# for age in all_ages:
#     print(age)  # leave to track while running
#     files = files_by_age.get(age, [])
#     for file in files:
#         wav_file = os.path.join('./TC_all/', file)
#         lengths_by_age[age] += get_wav_duration(wav_file)

# # print(lengths_by_age)

# # Extract keys and values
# ages = list(lengths_by_age.keys())
# values = list(lengths_by_age.values())
# divided_values = [value / 60 for value in values]

# plt.scatter(ages, divided_values, color='blue', marker='o')
# plt.xlabel('Age')
# plt.ylabel('Amount of data (minutes) after processing')
# # plt.title('Data (minutes) by age')
# plt.xticks(np.arange(20, 101, 10))
# plt.yticks(np.arange(0, 81, 10))
# plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/data_distr_all_2')

# plt.show()



# AGE AND DURATION, BUT ALSO BY SPEAKER
# use dict of dicts now
files_by_speaker_age = defaultdict(lambda: defaultdict(list))
lengths_by_speaker_age = defaultdict(lambda: defaultdict(float))

with open(audio_text_file, 'r') as file:
    for line in file:
        # like: wavs/queen_2008_3_1_007.wav|The impact of pollution falls unequally.|17|82
        info = line.split('|')
        age = int(info[-1])
        file_name = info[0]
        speaker = int(info[2])

        files_by_speaker_age[speaker][age].append(file_name)

for speaker, age_files in files_by_speaker_age.items():
    for age, files in age_files.items():
        print(speaker, age)  # leave to track while running
        for file in files:
            wav_file = os.path.join('./TC_all/', file)
            lengths_by_speaker_age[speaker][age] += get_wav_duration(wav_file)

# print(lengths_by_age)

speakers = ["reagan", "bowman", "byrne",  "cooke", "cronkite", "doyle", "dunne", "finucane", "gogan", "lawlor", "lockwood", "magee", "nibhriain", "odulaing", "plomley", "queen", "thatcher"]

plt.figure(figsize=(14, 10))

for i, (speaker, age_lengths) in enumerate(lengths_by_speaker_age.items()):

    # Extract keys and values
    ages = list(age_lengths.keys())
    # values = list(age_lengths[age])
    # divided_values = [value / 60 for value in values]
    durations = [age_lengths[age] / 60 for age in ages]
    speaker_name = speakers[speaker]
    plt.scatter(ages, durations, marker='o', label=speaker_name)


plt.xlabel('Age')
plt.ylabel('Amount of data (minutes) after processing, by speaker')
# plt.title('Data (minutes) by age')
plt.xticks(np.arange(20, 101, 10))
plt.yticks(np.arange(0, 71, 10))
plt.legend(bbox_to_anchor=(1.5, 0), loc='lower right', borderaxespad=0.)
plt.tight_layout()
plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/data_distr_all_spk')

plt.show()