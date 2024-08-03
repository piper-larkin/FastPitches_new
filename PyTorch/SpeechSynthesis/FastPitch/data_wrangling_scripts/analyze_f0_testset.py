import torch
import os
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt

## loads pitch files from some dir and gets avg f0
## for groups of files labeled with _age_spk at end
# like phrases30_35_3

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

    for age in range(18, 101):
        # print(age)
        sub_dir = directory_root + f'phrases30_{age}_{speaker}/pitch/'
        for file in os.listdir(sub_dir):
            if '.pt' in file:
                # print(file)
                age_pt_dict[age].append(os.path.join(sub_dir, file))
                break  # If a name is found, no need to check other names

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
    # ages = range(20, 101)

    for avg_f0_values, label in zip(avg_f0_dicts, labels):
        ages = sorted(avg_f0_values.keys())
        # avg_f0s = [avg_f0_values.get(age, np.nan) for age in ages]
        avg_f0s = [avg_f0_values[age] for age in ages]
        # plt.plot(ages, avg_f0s, marker='o', linestyle='-', label=label)   # for lines
        plt.scatter(ages, avg_f0s, label=label)

    plt.xlabel('Age')
    plt.ylabel('Average f0')
    plt.ylim(0, 401)
    plt.xlim(10, 101)
    plt.title('Average f0 for speech by age')   # Change
    plt.legend()
    plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/figures/f0_plot_18to100_nooutlier') # Change
    print('Figure saved')
    plt.show()

# vocoded_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_true_vocoded/pitch'
# synth_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_TC_inf/pitch'
# avg_f0_dicts = []
# age_f0_values = load_f0_values_age(vocoded_dir)
# # print(group, age_f0_values.keys())
# # print((sorted(age_f0_values.keys())))
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# # print(group, avg_f0s_age)
# avg_f0_dicts.append(avg_f0s_age)
# print(avg_f0s_age)
# labels = ['all']
# plot_avg_f0s_multi(avg_f0_dicts, labels)


# group1_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/pitch'
# group1_names = ['bowman']
# group2_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/pitch'
# group2_names = ['bowman_2006']

# speakers = ['bowman', 'byrne', 'cooke', 'cronkite', 'doyle', 'dunne', 'finucane', 'gogan', 'lawlor', 'lockwood', 'magee', 'nibhriain', 'odulaing', 'plomley', 'queen', 'thatcher', 'reagan']

# groups = ['cooke', 'queen', 'reagan']   
# labels = ['cooke', 'queen', 'reagan'] # labels for plotting, may differ if have groups like real/synth or m/f
# groups = ['bowman']
# avg_f0_dicts = []  # for plotting
# for group in groups:
#     print(group)
#     age_f0_values = load_f0_values_age(vocoded_dir, [group])
#     # age_f0_values = load_f0_values_age_specify(group1_dir, synth_dir, [group])
#     # print(group, age_f0_values.keys())
#     # print((sorted(age_f0_values.keys())))
#     avg_f0s_age = avg_f0_values_age(age_f0_values)
#     # print(group, avg_f0s_age)
#     avg_f0_dicts.append(avg_f0s_age)
#     # print(avg_f0s_age)
# plot_avg_f0s_multi(avg_f0_dicts, labels)


# labels = ['true', 'vocoded', 'synthetic']
# age_f0_values = load_f0_values_age_specify(group1_dir, synth_dir)
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# avg_f0_dicts.append(avg_f0s_age)

# age_f0_values = load_f0_values_age(vocoded_dir)
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# avg_f0_dicts.append(avg_f0s_age)

# age_f0_values = load_f0_values_age(synth_dir)
# avg_f0s_age = avg_f0_values_age(age_f0_values)
# avg_f0_dicts.append(avg_f0s_age)
# plot_avg_f0s_multi(avg_f0_dicts, labels)



# f0_values = load_f0_values_age(group1_dir, group1_names)
# print(f0_values)
# avg_f0 = avg_f0_values_age(f0_values)
# print(avg_f0)
# group1_avg_f0, group2_avg_f0 = compare_groups(group1_dir, group2_dir, group1_names, group2_names)
# print(group1_avg_f0, group2_avg_f0)


cooke_f0 = load_f0_values_age(directory_root, 3)
queen_f0 = load_f0_values_age(directory_root, 15)
reagan_f0 = load_f0_values_age(directory_root, 0)
# print(cooke_f0[27])
cooke_avgf0 = avg_f0_values_age(cooke_f0)
queen_avgf0 =  avg_f0_values_age(queen_f0)
reagan_avgf0 =  avg_f0_values_age(reagan_f0)

avg_f0_dicts = [cooke_avgf0, queen_avgf0, reagan_avgf0]

print(cooke_avgf0)
print(queen_avgf0)
print(reagan_avgf0)

labels = ['Cooke: synthesized speech', 'Queen: synthesized speech', 'Reagan: synthesized speech']
plot_avg_f0s_multi(avg_f0_dicts, labels)

