import torch
import os
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt

## loads pitch files from some dir and gets avg f0
##TODO add functionality to specify speaker(s)
##TODO add funtionality to plot

def load_f0_values(directory, names=None):
    '''
    names: list of speaker names want to consider
    returns list of f0 values, where each element is np.array

    '''
    pt_files = []
    f0_values = []
    for file in os.listdir(directory):
        if names is not None:
            for name in names:
                if name in file:
                    pt_files.append(os.path.join(directory, file))
                    break  # If a name is found, no need to check other names
        else:
            pt_files.append(os.path.join(directory, file))

    for pt_file in pt_files:
        f0_tensor = torch.load(pt_file)
        f0_values.append(f0_tensor.numpy())
    
    return f0_values

def avg_f0_values(f0_values):

    # Flatten the list of f0 values (from numpy arrays)
    flattened_f0_values = []
    for f0 in f0_values:
        flattened_array = f0.flatten() 
        flattened_f0_values.append(flattened_array)
    all_f0_values = np.concatenate(flattened_f0_values, axis=0)

    # Filter out zero values 
    # NOTE: confirm this is ok
    voiced_f0_values = all_f0_values[all_f0_values > 0]

    if len(voiced_f0_values) == 0:
        raise ValueError("No voiced frames found for averaging")
    
    average_f0 = np.mean(voiced_f0_values)
    return average_f0

def compare_groups(group1_dir, group2_dir, group1_names=None, group2_names=None):
    group1_f0_values = load_f0_values(group1_dir, group1_names)
    group2_f0_values = load_f0_values(group2_dir, group2_names)
    
    group1_avg_f0 = avg_f0_values(group1_f0_values)
    group2_avg_f0 = avg_f0_values(group2_f0_values)
    
    return group1_avg_f0, group2_avg_f0


def calculate_age(file):
    '''
    file: name of pt file
    returns age of spk in a single file 
    '''
    speaker_dict = {'bowman': (1942, 7, 28, 1),
                        'byrne': (1934, 8, 5, 2),
                        'cooke': (1908, 11, 20, 3),
                        'cronkite': (1916, 11, 4, 4),
                        'doyle': (1952, 1, 30, 5),
                        'dunne': (1958, 4, 28, 6),
                        'finucane': (1950, 5, 21, 7),
                        'gogan': (1934, 5, 3, 8),
                        'lawlor': (1961, 1, 1, 9),
                        'lockwood': (1916, 9, 5, 10),
                        'magee': (1935, 1, 31, 11),
                        'nibhriain': (1952, 1, 1, 12),
                        'odulaing': (1933, 3, 15, 13),
                        'plomley': (1914, 1, 20, 14),
                        'queen': (1926, 4, 21, 15),
                        'thatcher': (1925, 10, 13, 16),
                        'reagan': (1911, 2, 6, 0)}


    date_pattern = r'\d{4}-\d{2}-\d{2}|\d{4}_\d{2}_\d{2}|\d{4}'

    # will call this for each file 
    # split file just to get first part, which will be dict key
    parts = file.split('_')
    speaker = parts[0]
    (r_year, r_month, r_day, spk_id) = speaker_dict[speaker]
    # spk_id = str(spk_id)

    # then match date in whole file name
    date_match = re.search(date_pattern, file)
    date = date_match.group()

        
    if '_' in date:
        year, month, day = date.split('_')
        year = int(year)
        month = int(month)
        day = int(day)

        # Deal with months/days
        if month < r_month:  # before birth month
            age = year - r_year - 1
        elif month == r_month and day < r_day:  # birth month, before birthdate
            age = year - r_year - 1
        else:  # any date from birthdate onward
            age = year - r_year
    elif '-' in date:
        year, month, day = date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)

        # Deal with months/days
        if month < r_month:  # before birth month
            age = year - r_year - 1
        elif month == r_month and day < r_day:  # birth month, before birthdate
            age = year - r_year - 1
        else:  # any date from birthdate onward
            age = year - r_year
    else:
        year = int(date)
        age = year - r_year
    
    return age

def load_f0_values_age(directory, names=None):
    '''
    names: list of speaker names want to consider

    returns age_f0_values: dict containing list of f0 values by age, where each element is np.array

    '''
    age_pt_dict = defaultdict(list)
    age_f0_values = defaultdict(list)

    pt_files = []
    f0_values = []
    for file in os.listdir(directory):
        if names is not None:
            for name in names:
                if name in file:
                    age = calculate_age(file)
                    age_pt_dict[age].append(os.path.join(directory, file))
                    break  # If a name is found, no need to check other names
        else:
            age = calculate_age(file)
            age_pt_dict[age].append(os.path.join(directory, file))

    for age, pt_files in age_pt_dict.items():
        for pt_file in pt_files:
            f0_tensor = torch.load(pt_file)
            age_f0_values[age].append(f0_tensor.numpy())
    
    return age_f0_values

def load_f0_values_age_specify(directory, test_dir, names=None):
    '''
    names: list of speaker names want to consider

    returns age_f0_values: dict containing list of f0 values by age, where each element is np.array

    '''
    age_pt_dict = defaultdict(list)
    age_f0_values = defaultdict(list)

    pt_files = []
    f0_values = []

    for file in os.listdir(test_dir):
        if names is not None:
            for name in names:
                if name in file:
                    age = calculate_age(file)
                    age_pt_dict[age].append(os.path.join(directory, file))
                    break  # If a name is found, no need to check other names
        else:
            age = calculate_age(file)
            age_pt_dict[age].append(os.path.join(directory, file))

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
        
        avg_f0s_age[age] = np.mean(voiced_f0_values)
    
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
    plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/figures/f0_plot_all_spk2') # Change
    print('Figure saved')
    plt.show()

vocoded_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_true_vocoded/pitch'

synth_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_TC_inf/pitch'
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


group1_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/pitch'
group1_names = ['bowman']
group2_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/pitch'
group2_names = ['bowman_2006']

speakers = ['bowman', 'byrne', 'cooke', 'cronkite', 'doyle', 'dunne', 'finucane', 'gogan', 'lawlor', 'lockwood', 'magee', 'nibhriain', 'odulaing', 'plomley', 'queen', 'thatcher', 'reagan']

# groups = ['cooke', 'queen', 'reagan']   
# labels = ['cooke', 'queen', 'reagan'] # labels for plotting, may differ if have groups like real/synth or m/f
# groups = ['bowman']
avg_f0_dicts = []  # for plotting
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


labels = ['true', 'vocoded', 'synthetic']
age_f0_values = load_f0_values_age_specify(group1_dir, synth_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)

age_f0_values = load_f0_values_age(vocoded_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)

age_f0_values = load_f0_values_age(synth_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)
plot_avg_f0s_multi(avg_f0_dicts, labels)



# f0_values = load_f0_values_age(group1_dir, group1_names)
# print(f0_values)
# avg_f0 = avg_f0_values_age(f0_values)
# print(avg_f0)
# group1_avg_f0, group2_avg_f0 = compare_groups(group1_dir, group2_dir, group1_names, group2_names)
# print(group1_avg_f0, group2_avg_f0)

