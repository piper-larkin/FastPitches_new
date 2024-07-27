import random

# create all_lines like:
# find TC_for_fastpitch/filelists/*train.txt -type f -exec cat {} + > all_lines.txt

with open('TC_all/all_lines.txt', 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines to the output file
with open('TC_all/tc_LJ_spk_age_train.txt', 'w') as f:
    f.writelines(lines)