#!/usr/bin/env bash

set -e

: ${DATA_DIR:=80s_reagan_2} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/80s_reagan_2/reagan_audio_1980s_text_3.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
