from collections import defaultdict
import os

testset_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_test.txt'
phrases_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_phrases.txt'

lines_by_age = defaultdict(list)

# get all ages
with open(testset_file, 'r') as test_file:
    ages = set()
    for line in test_file:
        # mels/queen_1956_006.pt|pitch/queen_1956_006.pt|Happy Christmas from us all.|15|30
        split_line = line.split("|")
        age = split_line[-1]
        ages.add(age)
all_ages = list(ages)

# BELOW MAKES PHRASE SET PER SPK+AGE COMBO
for age in all_ages:
    with open(testset_file, 'r') as test_file:
        # add lines
        for line in test_file:
            # mels/queen_1956_006.pt|pitch/queen_1956_006.pt|Happy Christmas from us all.|15|30
            split_line = line.split("|")
            line_age = split_line[-1]
            line_spk = split_line[-2].strip()
            if line_age == age:
                line_age = split_line[-1].strip()
                phrases_file = f'/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_phrases_{line_spk}_{line_age}.tsv'
                
                if os.path.exists(phrases_file):
                    with open(phrases_file, 'a') as phrase_file:
                        wav_name = split_line[0][5:-3] + '.wav'
                        line_to_add = split_line[0] + '\t' + wav_name + '\t' + split_line[2] + '\n'
                        phrase_file.write(line_to_add)
                else:
                    with open(phrases_file, 'a') as phrase_file:
                        # add header
                        test_headers = ['mel', 'output', 'text']
                        headers = '\t'.join(test_headers)
                        phrase_file.write(headers + '\n')

                        wav_name = split_line[0][5:-3] + '.wav'
                        line_to_add = split_line[0] + '\t' + wav_name + '\t' + split_line[2] + '\n'
                        phrase_file.write(line_to_add)



# BELOW MAKES PHRASE SET PER AGE
# for age in all_ages:
#     phrases_file = f'/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/tc_audio_pitch_text_spk_age_phrases_{age}.txt'

#     with open(testset_file, 'r') as test_file:
#         with open(phrases_file, 'w') as phrase_file:
        
#             # add header
#             test_headers = ['mel', 'output', 'text']
#             headers = '\t'.join(test_headers)
#             phrase_file.write(headers + '\n')

#             # add lines
#             for line in test_file:
#                 # mels/queen_1956_006.pt|pitch/queen_1956_006.pt|Happy Christmas from us all.|15|30
#                 split_line = line.split("|")
#                 line_age = split_line[-1]
#                 if line_age == age:
#                     wav_name = split_line[0][5:-3] + 'wav'
#                     line_to_add = split_line[0] + '\t' + wav_name + '\t' + split_line[2] + '\n'
#                     phrase_file.write(line_to_add)


