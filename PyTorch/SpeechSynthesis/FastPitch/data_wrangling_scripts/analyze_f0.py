import torch
import os
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import linregress

'''
Overview:
Loads pitch files some directory and gets average f0
Get age information to compare f0s by age
Plot avg f0s
'''



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
                    break  
        else:
            pt_files.append(os.path.join(directory, file))

    for pt_file in pt_files:
        f0_tensor = torch.load(pt_file)
        f0_values.append(f0_tensor.numpy())
    
    return f0_values

def avg_f0_values(f0_values):
    '''
    f0_values: all values from load_f0_values function
    returns average f0 value
    '''

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
        
    average_f0 = np.mean(filtered_f0s)
    return average_f0

def compare_groups(group1_dir, group2_dir, group1_names=None, group2_names=None):
    '''
    group1_dir: directory path for pitch files of group 1
    group1_names: list of names to use for group 1, or None
    group2_dir: directory path for pitch files of group 2
    group2_names: list of names to use for group 2, or None

    Returns f0 values for 2 groups
    '''

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

    # split file just to get first part, which will be dict key
    if 'LJ' not in file:
        parts = file.split('_')
        speaker = parts[0]
        (r_year, r_month, r_day, spk_id) = speaker_dict[speaker]

    # then match date in whole file name
    date_match = re.search(date_pattern, file)
    date = date_match.group()

    if 'LJ' in file:
        age = 45
        
    elif '_' in date:
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
                    break 
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
                    break 
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
    age_f0_values: dict of all f0 values by age
    returns dict, with avg per age
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
    
    Creates plot
    '''

    for avg_f0_values, label in zip(avg_f0_dicts, labels):
        if 'female' in label:
            color = 'gold'
        else:
            color = 'green'
        ages = sorted(avg_f0_values.keys())
        # avg_f0s = [avg_f0_values.get(age, np.nan) for age in ages]
        avg_f0s = [avg_f0_values[age] for age in ages]
        # plt.plot(ages, avg_f0s, marker='o', linestyle='-', label=label)   # for lines
        plt.scatter(ages, avg_f0s, label=label, color=color)

        # Calculate the trend line using linear regression
        slope, intercept, r_value, p_value, std_err = linregress(ages, avg_f0s)
        trend_line = slope * np.array(ages) + intercept
        
        # Plot the trend line
        plt.plot(ages, trend_line, linestyle='-', color=color)

    plt.xlabel('Chronological age (years)')
    plt.ylabel('Average F0 (Hz)')
    plt.ylim(0, 350)
    plt.xlim(10, 101)
    # plt.title('Average f0 for speech by age')   # Change
    plt.legend()
    # plt.savefig('/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/figures/f0_male_vs_female_line') # Change
    # print('Figure saved')
    plt.show()


vocoded_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_true_vocoded/pitch'
synth_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/testset_TC_inf/pitch'
data_dir = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/pitch'


# # Sample use comparing avg F0 of male and female speakers
# male_speakers = ['bowman', 'byrne', 'cooke', 'cronkite', 'gogan', 'magee', 'odulaing', 'plomley', 'reagan']
# female_speakers = ['doyle', 'dunne', 'finucane', 'lawlor', 'lockwood', 'nibhriain', 'thatcher', 'queen']

# male_dict = load_f0_values_age(data_dir, male_speakers)
# male_dict_avg = avg_f0_values_age(male_dict)
# female_dict = load_f0_values_age(data_dir, female_speakers)
# female_dict_avg = avg_f0_values_age(female_dict)
# plot_avg_f0s_multi([male_dict_avg, female_dict_avg], ['male speech', 'female speech'])


# Sample use comparing true, vocoded, and synthetic speech 
labels = ['true', 'vocoded', 'synthetic']
age_f0_values = load_f0_values_age_specify(data_dir, synth_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)

age_f0_values = load_f0_values_age(vocoded_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)

age_f0_values = load_f0_values_age(synth_dir)
avg_f0s_age = avg_f0_values_age(age_f0_values)
avg_f0_dicts.append(avg_f0s_age)
plot_avg_f0s_multi(avg_f0_dicts, labels)
