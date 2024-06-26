import os
import wave
import contextlib
import matplotlib.pyplot as plt
import numpy as np



def get_wav_duration(wav_file):
    """Return the duration of a WAV file in seconds."""
    with contextlib.closing(wave.open(wav_file, 'r')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration

def get_all_wav_durations(folder_path):
    """Return a list of durations for all WAV files in the given folder."""
    wav_durations = []
    dur_over30 = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            wav_path = os.path.join(folder_path, filename)
            duration = get_wav_duration(wav_path)
            if duration > 30:
                dur_over30.append(filename)
            wav_durations.append(duration)
    return wav_durations, dur_over30

def plot_duration_histogram(wav_durations):
    """Plot a histogram of WAV file durations."""
    plt.hist(wav_durations, bins=len(wav_durations)//10, edgecolor='black')
    plt.xlim(min(wav_durations), max(wav_durations))
    plt.title('Histogram of WAV File Durations (LJS)')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    # plt.xticks(np.arange(min(wav_durations), max(wav_durations) + 1, 5))  # Adjust step size as needed
    plt.xticks(np.arange(min(wav_durations), max(wav_durations) + 1, 1), fontsize=6, rotation=70, ha='right')
    plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/data_wrangling_scripts/histogram_LJS_sec')
    plt.show()

# Path to the folder containing WAV files
# folder_path = '/Users/piperlarkin/Downloads/Dissertation materials/TCDSA/main/Data/reagan_80s_sentence_wavs'
folder_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs'

# Get durations of all WAV files in the folder
wav_durations, dur_over30 = get_all_wav_durations(folder_path)

print(len(dur_over30))
# print(dur_over30)
# Plot the histogram of durations
plot_duration_histogram(wav_durations)

over_30 = 0
over_60 = 0
over_15 = 0
print(len(wav_durations))

for dur in wav_durations:
    if dur > 30:
        over_30 += 1
    if dur > 60:
        over_60 += 1
    if dur > 15:
        over_15 += 1
print(over_60)
print(over_30)
print(over_15)

print(max(wav_durations))
print(min(wav_durations))
# 0
# 13100
# 0
# 0
# 0
# 10.096190476190475
# 1.1100680272108843
