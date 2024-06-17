#!/usr/bin/env bash

set -e

: ${DATA_DIR:=audio} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/test_audio_text.txt \
    --n-workers 0 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
