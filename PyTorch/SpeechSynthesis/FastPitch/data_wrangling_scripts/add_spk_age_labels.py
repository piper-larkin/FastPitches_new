# to read in LJ train_v3 and val files (audio_pitch versions)
# and create new file which has |17|45 at end of each line 

train_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_train_v3.txt'
dev_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_val.txt'

train_new = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_spk_age_train.txt'
dev_new = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_spk_age_val.txt'


def add_spk_age(orig_file, new_file):
    new_lines = []
    with open(orig_file, 'r') as orig_file:
        for line in orig_file:
            line = line.strip('\n')
            new_line = line + '|17|45\n'  # hard coded to LJ
            new_lines.append(new_line)
        with open(new_file, 'w') as new_file:
            new_file.writelines(new_lines)

    print('new file complete')

add_spk_age(train_file, train_new)
add_spk_age(dev_file, dev_new)
            

