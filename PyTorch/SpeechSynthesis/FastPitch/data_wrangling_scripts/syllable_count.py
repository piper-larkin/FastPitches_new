'''
Comparing speaker rate in real vs synthesized data
Syllable (vowel) per second
Compare for various ages
'''

import librosa
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Path to all files
# format: wavs/reagan_1986_28_1_002.wav|Today is a day for morning and remembering.|1|75
file_list_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/reagan_all/reagan_audio_text_age_spk.txt'
# Path to synthesized sentences
# format: mels/reagan_1989_11_11_036.pt	reagan_1989_11_11_036.wav	And at the end, together we're reaching our destination.
# NOTE: need to skip header
synth_inference_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/phrases/testset_1to30_80s.tsv'


def count_vowels(sentence):
    # regex for finding vowels
    vowels = r'[aeiouAEIOU]'
    matches = re.findall(vowels, sentence)
    return len(matches)

# Get file names in synth devset
def synth_rate(synth_inference_file, age):
    '''
    Need to run each time for each age
    '''
    total_duration_synth = 0.0
    v_count = 0
    with open(synth_inference_file, 'r') as file:
        next(file)
        for line in file:
            columns = line.strip().split('\t')

            # Get length synth file
            label = columns[1]
            # NOTE: may have to change below depending on how inf output is named
            wav_file_path = f'/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/reagan_deep_age/audio_testset_1to30_80s.tsv_{age}/{label}'
            y, sr = librosa.load(wav_file_path)
            total_duration_synth += librosa.get_duration(y, sr)

            # Get num vowels in the line
            sentence = columns[2]
            v_count += count_vowels(sentence)
    return float(v_count / total_duration_synth)
            
def real_rate(file_list_path):
    '''
    Run once for all ages
    '''
    v_count_dict = defaultdict(float)
    dur_dict = defaultdict(float)
    rate_dict_real = defaultdict(float)

    with open(file_list_path, 'r') as file:
        for line in file:
            dur = 0.0
            v_count = 0
            columns = line.strip().split('|')
            age = columns[3]
            sentence = columns[1]
            wav_file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/reagan_all/' + columns[0]
            y, sr = librosa.load(wav_file_path)
            dur = librosa.get_duration(y, sr)
            v_count = count_vowels(sentence)

            # need to get total dur and total vowels per age. then divide at end
            v_count_dict[age] += v_count
            dur_dict[age] += dur

        for age in list(dur_dict.keys()):
            rate_dict_real[int(age)] = float(v_count_dict[age] / dur_dict[age])

    return rate_dict_real

def plot_rates(rate_dict_real, rate_dict_synth):
    '''
    Use dicts from other funcs
    Plot speech rate by age 
    '''
    # Extract data
    real_x = list(rate_dict_real.keys())
    real_y = list(rate_dict_real.values())

    synth_x = list(rate_dict_synth.keys())
    synth_y = list(rate_dict_synth.values())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(real_x, real_y, marker='o', label='True speech')
    plt.plot(synth_x, synth_y, marker='s', label='Synthesized speech')

    plt.title('Syllables per second of real vs synthetic speech by age (Reagan)')
    plt.xlabel('Age (years)')
    plt.ylabel('Speech rate (syllable/second)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reagan_deep_age/plots/speech_rate1.png')
    plt.show()



# Add rates to synth dict
rate_dict_synth = defaultdict(int)
ages = [37, 69, 83]    # NOTE need to change each time
for age in ages:
    synth_v_rate = synth_rate(synth_inference_file, age)
    rate_dict_synth[age] += synth_v_rate

# Rate of actual wavs
rate_dict_real = real_rate(file_list_path)

plot_rates(rate_dict_real, rate_dict_synth)

# print('synth: ', rate_dict_synth)
# print('real: ', rate_dict_real)

# {37: 6.518922738693466, 69: 5.087316176470589, 83: 4.418295379509279}
# {'75': 4.8454584892181565, '77': 4.545497972020187, '71': 4.817487303455395, '72': 5.00361976196841, '70': 4.846913947965634, '76': 4.639965769630594, '74': 4.7571079537746135, '81': 3.8630265640765193, '73': 4.783589315832243, '78': 4.7136182442273205, '83': 4.453456282249981, '53': 5.287050963151505, '37': 5.476623787289584, '63': 5.151984619147958, '50': 5.458152525416166, '43': 5.781272581245099}