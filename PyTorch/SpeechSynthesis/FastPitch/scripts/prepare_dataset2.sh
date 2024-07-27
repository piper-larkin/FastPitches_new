#!/usr/bin/env bash

set -e

: ${DATA_DIR:=TC_all/testset_true_vocoded} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists TC_all/tc_audio_text_spk_age_test.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    --n-speakers 17 \
    $ARGS
