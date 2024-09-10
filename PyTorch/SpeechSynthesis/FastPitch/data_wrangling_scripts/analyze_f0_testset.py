import torch
import os
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt

'''
Overview:
Loads pitch files from some dir and gets avg f0 for groups of files labeled with _age_spk at end
like phrases30_35_3
'''

synth_inference_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_phrases_30.tsv'
directory_root = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/graphemes_out_all/'

def load_f0_values_age(directory, speaker=None):
    '''
    directory = directory_root 
    speaker = spk id (int)
    returns age_f0_values: dict containing list of f0 values by age, where each element is np.array

    '''
    age_pt_dict = defaultdict(list)
    age_f0_values = defaultdict(list)

    pt_files = []
    f0_values = []

    for age in range(20, 101, 5):
        sub_dir = directory_root + f'phrases30_{age}_{speaker}/pitch/'
        for file in os.listdir(sub_dir):
            if '.pt' in file:
                age_pt_dict[age].append(os.path.join(sub_dir, file))
                break  

    for age, pt_files in age_pt_dict.items():
        for pt_file in pt_files:
            f0_tensor = torch.load(pt_file)
            age_f0_values[age].append(f0_tensor.numpy())
    
    return age_f0_values


def avg_f0_values_age(age_f0_values):
    '''
    return dict, with avg per age
    '''

    avg_f0s_age = defaultdict(float)

    for age, f0_values in age_f0_values.items():
        # Flatten the list of f0 values (from numpy arrays)
        flattened_f0_values = []
        for f0 in f0_values:
            flattened_array = f0.flatten() 
            flattened_f0_values.append(flattened_array)
        all_f0_values = np.concatenate(flattened_f0_values, axis=0)

        # Filter out zero values 
        voiced_f0_values = all_f0_values[all_f0_values > 0]

        if len(voiced_f0_values) == 0:
            raise ValueError("No voiced frames found for averaging")
        
        # Remove outliers, based on IQR
        Q1 = np.percentile(voiced_f0_values, 25)
        Q3 = np.percentile(voiced_f0_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_f0s = voiced_f0_values[(voiced_f0_values >= lower_bound) & (voiced_f0_values <= upper_bound)]
        
        avg_f0s_age[age] = np.mean(filtered_f0s)
    
    sorted_avg_f0s_age = dict(sorted(avg_f0s_age.items()))
    return sorted_avg_f0s_age
    

def plot_avg_f0s_multi(avg_f0_dicts, labels):
    '''
    avg_f0_dicts: list of dicts by age, one per group 
    labels: list of labels for plotting, label is by group
    '''

    for avg_f0_values, label in zip(avg_f0_dicts, labels):
        # Change color conditions if needed 
        if 'synth' in label:
            color = 'red'
        elif 'true speech' in label:
            color = '#9467BD'
        else:
            color = 'gold'
        ages = sorted(avg_f0_values.keys())
        avg_f0s = [avg_f0_values[age] for age in ages]
        plt.scatter(ages, avg_f0s, label=label, color=color)

    plt.xlabel('Chronological Age (years)')
    plt.ylabel('Average F0 (Hz)')
    plt.ylim(0, 351)
    plt.xlim(10, 101)
    plt.legend()
    # plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/figures/f0_comparison') # Change
    # print('Figure saved')
    plt.show()



# Sample use comparing f0s for Queen's true and synthetic speech to known avg values for all female speakers
avg_f0_dicts = [{19: 203.6071, 22: 199.55142, 24: 196.13388, 26: 299.31296, 28: 201.96864, 29: 225.50632, 30: 260.6254, 31: 184.00262, 32: 264.24564, 33: 249.98218, 34: 221.24025, 35: 194.17413, 37: 191.23828, 38: 226.35803, 39: 213.74268, 40: 230.58405, 41: 224.50577, 42: 236.87819, 44: 202.93994, 45: 239.20944, 46: 191.8246, 48: 196.8232, 50: 249.42766, 51: 172.44994, 54: 257.25247, 55: 278.0121, 56: 126.73211, 57: 202.0656, 58: 180.85393, 59: 220.84906, 60: 272.8948, 62: 229.34761, 63: 183.20555, 64: 143.18718, 65: 219.64798, 67: 260.5924, 68: 227.75212, 69: 239.00542, 70: 187.65216, 71: 220.9069, 72: 218.77022, 73: 227.03705, 74: 220.66692, 75: 212.97124, 76: 214.70087, 80: 238.44916, 81: 209.46542, 82: 228.80234, 83: 213.47461, 84: 193.30183},
{26: 299.31296, 28: 279.89053, 29: 263.11288, 30: 260.6254, 32: 264.24564, 33: 249.98218, 34: 259.79065, 35: 244.61348, 37: 237.89578, 38: 226.35803, 39: 241.43674, 40: 230.58405, 41: 259.20486, 42: 236.87819, 44: 234.41621, 45: 239.20944, 46: 231.4634, 57: 230.81761, 59: 220.84906, 62: 234.77148, 68: 227.75212, 69: 239.00542, 70: 227.93077, 71: 220.9069, 72: 218.77022, 73: 227.03705, 74: 220.66692, 75: 212.97124, 76: 214.70087, 80: 238.44916, 81: 206.33743, 82: 228.80234, 83: 213.47461, 84: 209.98846},
]
f0_values = load_f0_values_age(directory_root, 15)
avg_f0 = avg_f0_values_age(f0_values)
avg_f0_dicts.append(avg_f0)
labels = ['Average true female speech', 'Queen: true speech', 'Queen: synthetic speech']
plot_avg_f0s_multi(avg_f0_dicts, labels)


# # Sample use comapring f0 values of true and synthetic speech of two speakers to known avg of all female speakers in the dataset
# dir_root1 = 'TC_all/lawlor_rainbow/'
# dir_root2 = 'TC_all/lj_rainbow/'
# avg_f0_dicts = [{19: 203.6071, 22: 199.55142, 24: 196.13388, 26: 299.31296, 28: 201.96864, 29: 225.50632, 30: 260.6254, 31: 184.00262, 32: 264.24564, 33: 249.98218, 34: 221.24025, 35: 194.17413, 37: 191.23828, 38: 226.35803, 39: 213.74268, 40: 230.58405, 41: 224.50577, 42: 236.87819, 44: 202.93994, 45: 239.20944, 46: 191.8246, 48: 196.8232, 50: 249.42766, 51: 172.44994, 54: 257.25247, 55: 278.0121, 56: 126.73211, 57: 202.0656, 58: 180.85393, 59: 220.84906, 60: 272.8948, 62: 229.34761, 63: 183.20555, 64: 143.18718, 65: 219.64798, 67: 260.5924, 68: 227.75212, 69: 239.00542, 70: 187.65216, 71: 220.9069, 72: 218.77022, 73: 227.03705, 74: 220.66692, 75: 212.97124, 76: 214.70087, 80: 238.44916, 81: 209.46542, 82: 228.80234, 83: 213.47461, 84: 193.30183},
# {24: 196.13388, 34: 203.44371, 44: 166.1424}, {45: 219.23178}
# ]
# age_f0_values = load_f0_values_age(dir_root1, 9)
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# print(avg_f0s_age)
# avg_f0_dicts.append(avg_f0s_age)
# age_f0_values = load_f0_values_age(dir_root2, 17)
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# avg_f0_dicts.append(avg_f0s_age)

# labels = ['Average true female speech', 'Lawlor: true speech', 'LJ: true speech', 'Lawlor: synthetic speech', 'LJ: synthetic speech']
# plot_avg_f0s_multi(avg_f0_dicts, labels)



# # Sample use comparing 3 speakers separately 
# cooke_f0 = load_f0_values_age(directory_root, 3)
# queen_f0 = load_f0_values_age(directory_root, 15)
# reagan_f0 = load_f0_values_age(directory_root, 0)

# cooke_avgf0 = avg_f0_values_age(cooke_f0)
# queen_avgf0 =  avg_f0_values_age(queen_f0)
# reagan_avgf0 =  avg_f0_values_age(reagan_f0)

# avg_f0_dicts = [cooke_avgf0, queen_avgf0, reagan_avgf0]

# labels = ['Cooke: synthesized speech', 'Queen: synthesized speech', 'Reagan: synthesized speech']
# plot_avg_f0s_multi(avg_f0_dicts, labels)

