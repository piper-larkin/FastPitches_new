import librosa 
import os

file_list_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_train_v3.txt'
# new_train_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_train_v3_small.txt'
new_train_file = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_pitch_text_val_small.txt'

total_duration = 0.0
# reagan_train_dur = 16121.245714285695   # seconds
reagan_dev_dur = 1082.9761904761908


with open(file_list_path, 'r') as file, open(new_train_file, 'a') as out_file:
    for line in file:
        # line like: mels/LJ050-0234.pt|pitch/LJ050-0234.pt|It has used other ...
        columns = line.strip().split('|')
        file_name = columns[0]
        label = file_name[5:-3]
        print(label)
        
        file_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/LJSpeech-1.1/wavs/' + label + '.wav'

        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y, sr)

        # Write line to a shorter training file list
        if total_duration + duration <= reagan_dev_dur:        # CHANGE MAX DURATION HERE
            out_file.write(line)
            total_duration += duration
            print(total_duration)
        else:
            break

    print(f"Total duration of new file: {total_duration} seconds")
