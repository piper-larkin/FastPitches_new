#!/usr/bin/env bash

set -e

for AGE in $(seq 18 100); do
    for SPEAKER in 0 3 15; do

        DATA_DIR="./TC_all/graphemes_out_all/phrases30_${AGE}_${SPEAKER}" # CHANGED
        # Create the 'mels' directory within DATA_DIR
         # Create the 'wavs' directory within DIR
        mkdir -p "$DATA_DIR/pitch"
        # WAVS_DIR="$DATA_DIR/wavs"
        # mkdir -p "$WAVS_DIR"
        

        # # Move all files and folders (excluding the 'wavs' directory itself) into the 'wavs' directory
        # for item in "$DATA_DIR"/*; do
        #     if [ "$(basename "$item")" != "wavs" ] && [ "$basename_item" != "mels" ]; then
        #         mv "$item" "$WAVS_DIR"
        #     fi
        # done

        # : ${ARGS="--extract-mels"}

        python prepare_dataset.py \
            --wav-text-filelists TC_all/tc_audio_text_spk_age_test30.txt \
            --n-workers 16 \
            --batch-size 1 \
            --dataset-path $DATA_DIR \
            --extract-pitch \
            --f0-method pyin \
            --n-speakers 17 \
            $ARGS
    done
done