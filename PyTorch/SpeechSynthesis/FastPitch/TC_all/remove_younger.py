# remove cooke lines earlier than 1965 from datasets

lines_keep = []
with open('TC_all/tc_audio_pitch_text_spk_age_dev.txt', 'r') as f:
    to_omit = ['1947', '1951', '1953', '1960', '1962'] 

    for line in f:
        columns = line.split('|')
        if not any(date in columns[0] for date in to_omit):
            lines_keep.append(line)


# Write the shuffled lines to the output file
with open('TC_all/tc_less_cooke_dev.txt', 'w') as f:
    for line in lines_keep:
        f.write(line)