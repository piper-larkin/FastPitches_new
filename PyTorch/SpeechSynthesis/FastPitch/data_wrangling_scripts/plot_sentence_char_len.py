import os
import wave
import contextlib
import matplotlib.pyplot as plt
import numpy as np

# sentence_file = '/Users/piperlarkin/Downloads/Dissertation materials/Whisper_outputs/reagan_sentences_5.tsv'
sentence_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_text.txt'

def get_char_len(sentence_file):
    """Return the length of sentences in # characters."""
    sentence_lengths = []
    over_256 = []

    with open(sentence_file, 'r') as sent_file:
        for line in sent_file:
            # columns = line.strip().split('\t')
            columns = line.strip().split('|')
            label = columns[0]
            sentence = columns[1]
            sentence_lengths.append(len(sentence))
            if len(sentence) > 256:
                over_256.append(label)
    return sentence_lengths, over_256


def plot_duration_histogram(sentence_lengths):
    """Plot a histogram of WAV file durations."""
    plt.hist(sentence_lengths, bins=len(sentence_lengths)//20, edgecolor='black')
    # plt.xlim(0, max(sentence_lengths))
    plt.xlim(0, max(sentence_lengths))

    plt.title('Histogram of Sentence Lengths (LJS)')
    plt.xlabel('Character count')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, max(sentence_lengths) + 1, 5), fontsize=6, rotation=70, ha='right')
    plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/data_wrangling_scripts/histogram_LJS_char')
    plt.show()


# Get durations of all WAV files in the folder
sent_lengths, over_256 = get_char_len(sentence_file)
# print(over_256)
# print(len(over_256))
# Plot the histogram of durations
plot_duration_histogram(sent_lengths)

print(len(sent_lengths))

over_300 = 0
over_128 = 0
over_200 = 0
for count in sent_lengths:
    if count > 300:
        over_300 +=1
    if count > 128:
        over_128 +=1
    if count > 200:
        over_200 +=1
print(over_128)
print(over_200)
print(len(over_256))
print(over_300)

print(max(sent_lengths))
print(min(sent_lengths))

# 13100
# 2981
# 0
# 0
# 0
# 187
# 12