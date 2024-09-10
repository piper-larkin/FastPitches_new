'''
Comparing speaker rate in real vs synthesized data
Syllable (vowel) per second
Compare for various ages
'''

import librosa
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
## for groups of files labeled with _age_spk at end
# like phrases30_35_3

# Path to all files
test_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_test.txt'
# test_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/reagan_all/reagan_audio_pitch_text_test_age_spk.txt'
wav_dir_path_real = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/'
wav_dir_path_synth1 = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/reagan_deep_age/testset_inf/'
# wav_dir_path_synth2 = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/reagan_all_3_linear/testset_inf/'
wav_dir_path_synth2 = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/graphemes_out_all/reagan_phrases/'

synth_inference_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_phrases_30.tsv'

def count_vowels(sentence):
    # regex for finding vowels
    vowels = r'[aeiouAEIOU]'
    matches = re.findall(vowels, sentence)
    return len(matches)
            
def get_rates(file_list_path, wav_dir_path):
    '''
    Run once for all ages
    '''
    v_count_dict = defaultdict(float)
    dur_dict = defaultdict(float)
    rate_dict_real = defaultdict(float)
    # mels/plomley_1958_008.pt|pitch/plomley_1958_008.pt|here is Miss Agnes Nichols|14|44
    with open(file_list_path, 'r') as file:
        for line in file:
            dur = 0.0
            v_count = 0
            columns = line.strip().split('|')
            age = columns[-1]
            sentence = columns[2]
            wav_file = columns[0][5:-3] + '.wav'
            # print(wav_file)
            wav_file_path = wav_dir_path + wav_file  # NOTE: WILL NEED TO CHANGE FOR INF/VOCODED
            y, sr = librosa.load(wav_file_path)
            dur = librosa.get_duration(y, sr)
            v_count = count_vowels(sentence)

            # need to get total dur and total vowels per age. then divide at end
            v_count_dict[age] += v_count
            dur_dict[age] += dur

        for age in list(dur_dict.keys()):
            rate_dict_real[int(age)] = float(v_count_dict[age] / dur_dict[age])

    # Sort the rate_dict_real by keys and create a new sorted dictionary
    sorted_rate_dict_real = {k: rate_dict_real[k] for k in sorted(rate_dict_real.keys())}

    return sorted_rate_dict_real

def get_rates_spk(file_list_path, wav_dir_path, speaker):
    '''
    Run once for all ages
    '''
    v_count_dict = defaultdict(float)
    dur_dict = defaultdict(float)
    rate_dict_real = defaultdict(float)
    # mels/plomley_1958_008.pt|pitch/plomley_1958_008.pt|here is Miss Agnes Nichols|14|44
    with open(file_list_path, 'r') as file:
        for line in file:
            columns = line.strip().split('|')
            if speaker in columns[0]:   # added for single speaker implementation
                dur = 0.0
                v_count = 0
                age = columns[-1]
                sentence = columns[2]
                wav_file = columns[0][5:-3] + '.wav'
                # print(wav_file)
                wav_file_path = wav_dir_path + wav_file  
                y, sr = librosa.load(wav_file_path)
                dur = librosa.get_duration(y, sr)
                v_count = count_vowels(sentence)

                # need to get total dur and total vowels per age. then divide at end
                v_count_dict[age] += v_count
                dur_dict[age] += dur

        for age in list(dur_dict.keys()):
            rate_dict_real[int(age)] = float(v_count_dict[age] / dur_dict[age])

    # Sort the rate_dict_real by keys and create a new sorted dictionary
    sorted_rate_dict_real = {k: rate_dict_real[k] for k in sorted(rate_dict_real.keys())}

    return sorted_rate_dict_real

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

    plt.figure(figsize=(10, 5))
    plt.ylim(2.1, 5.9)
    plt.plot(real_x, real_y, marker='*', color = '#9467BD', label='Queen: True speech')
    plt.plot(synth_x, synth_y, marker='.', color = 'red', label='Queen: Synthesized speech')

    # Fit a linear trend line to the real data
    z = np.polyfit(real_x, real_y, 2)  # 1 is the degree of the polynomial (linear)
    p = np.poly1d(z)  # Create a polynomial function from the coefficients
    plt.plot(real_x, p(real_x), color='grey', linestyle='--', label='Trend in true speaking rate')



    # plt.title('Syllables per second of real vs synthetic speech by age')   # CHANGE
    plt.xlabel('Chronological Age (years)')
    plt.ylabel('Speech rate (syllable/second)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/sr_queen_real_vs_synth2.png')
    plt.show()

def plot_rates_3(rate_dict_real, rate_dict_synth1, rate_dict_synth2):
    '''
    Use dicts from other funcs
    Plot speech rate by age 
    '''
    # Extract data
    real_x = list(rate_dict_real.keys())
    real_y = list(rate_dict_real.values())

    synth_x1 = list(rate_dict_synth1.keys())
    synth_y1 = list(rate_dict_synth1.values())

    synth_x2 = list(rate_dict_synth2.keys())
    synth_y2 = list(rate_dict_synth2.values())

    # Plot
    plt.figure(figsize=(10, 5))
    # plt.plot(real_x, real_y, marker='s', label='Cooke: synthesized speech')
    # plt.plot(synth_x1, synth_y1, marker='o', label='Queen: synthesized speech')  # CHANGED ORDER
    # plt.plot(synth_x2, synth_y2, marker='*', label='Reagan: synthesized speech')

    plt.plot(real_x, real_y, marker='s', label='reagan')
    plt.plot(synth_x1, synth_y1, marker='o', label='cooke')  # CHANGED ORDER
    plt.plot(synth_x2, synth_y2, marker='*', label='bowman')

    # plt.title('Syllables per second of real vs synthetic speech by age')   # CHANGE
    plt.xlabel('Age (years)')
    # yticks = np.arange(0, 12, 1)  
    yticks = np.arange(0, 9, 1)  
    plt.yticks(yticks)
    # plt.ylim(0, 12)
    plt.ylim(0, 9)
    plt.ylabel('Speech rate (syllable/second)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/rates_examine_5.png')
    plt.show()


def synth_rate_range(synth_inference_file, age, speaker):
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
            wav_file_path = f'/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/graphemes_out_all/phrases30_{age}_{speaker}/wavs/{label}'
            # wav_file_path = f'/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/lj_rainbow/rainbow_{age}_{speaker}/wavs/{label}'

            y, sr = librosa.load(wav_file_path)
            total_duration_synth += librosa.get_duration(y, sr)

            # Get num vowels in the line
            sentence = columns[2]
            v_count += count_vowels(sentence)
    return float(v_count / total_duration_synth)
# Rate of actual wavs
# rate_dict_real = get_rates(test_file, wav_dir_path_real)
# rate_dict_synth1 = get_rates(test_file, wav_dir_path_synth1)
# rate_dict_synth2 = get_rates(test_file, wav_dir_path_synth2)
# # plot_rates(rate_dict_real, rate_dict_synth)
# plot_rates_3(rate_dict_real, rate_dict_synth1, rate_dict_synth2)

# print('synth deep: ', rate_dict_synth1)
# print('synth tc on reagan phrases: ', rate_dict_synth2)
# print('real: ', rate_dict_real)

# # Real times for qcr 
# queen_dict = get_rates_spk(test_file, wav_dir_path_real, 'queen')
# cooke_dict = get_rates_spk(test_file, wav_dir_path_real, 'cooke')
# reagan_dict = get_rates_spk(test_file, wav_dir_path_real, 'reagan')
# print('queen rates: ', queen_dict)
# print('cooke rates: ', cooke_dict)
# print('reagan rates: ', reagan_dict)

# plot_rates_3(queen_dict, cooke_dict, reagan_dict)

# Add rates to synth dict
# rate_dict_synth_cooke = defaultdict(int)
# ages = range(18, 101)   
# for age in ages:
#     synth_v_rate = synth_rate_range(synth_inference_file, age, 3)
#     rate_dict_synth_cooke[age] += synth_v_rate

# rate_dict_synth_queen = defaultdict(int)
# ages = range(18, 101)     
# for age in ages:
#     synth_v_rate = synth_rate_range(synth_inference_file, age, 15)
#     rate_dict_synth_queen[age] += synth_v_rate

# rate_dict_synth_reagan = defaultdict(int)
# ages = range(18, 101)     
# for age in ages:
#     synth_v_rate = synth_rate_range(synth_inference_file, age, 0)
#     rate_dict_synth_reagan[age] += synth_v_rate

# plot_rates_3(rate_dict_synth_cooke, rate_dict_synth_queen, rate_dict_synth_reagan)

# print('cooke dict 18 to 100: ', rate_dict_synth_cooke)
# print('queen dict 18 to 100: ', rate_dict_synth_queen)
# print('reagan dict 18 to 100: ', rate_dict_synth_reagan)
# From reagan:
# {37: 6.518922738693466, 69: 5.087316176470589, 83: 4.418295379509279}
# {'75': 4.8454584892181565, '77': 4.545497972020187, '71': 4.817487303455395, '72': 5.00361976196841, '70': 4.846913947965634, '76': 4.639965769630594, '74': 4.7571079537746135, '81': 3.8630265640765193, '73': 4.783589315832243, '78': 4.7136182442273205, '83': 4.453456282249981, '53': 5.287050963151505, '37': 5.476623787289584, '63': 5.151984619147958, '50': 5.458152525416166, '43': 5.781272581245099}



# get_rates_spk(file_list_path, wav_dir_path, speaker)

# dict1 = get_rates_spk(test_file, wav_dir_path_real, 'reagan')
# dict2 = get_rates_spk(test_file, wav_dir_path_real, 'cooke')
# dict3 = get_rates_spk(test_file, wav_dir_path_real, 'bowman')

# plot_rates_3(dict1, dict2, dict3)

# synth_inference_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/phrases/rainbow_passage_etc_phrases.tsv'

rate_queen_synth = defaultdict(int)
ages = range(20, 101, 5)     
for age in ages:
    synth_v_rate = synth_rate_range(synth_inference_file, age, 15)
    rate_queen_synth[age] += synth_v_rate

queen_real = get_rates_spk(test_file, wav_dir_path_real, 'queen')

plot_rates(queen_real, rate_queen_synth)
