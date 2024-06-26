#!/usr/bin/env bash

set -e

: ${DATA_DIR:=80s_reagan} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/80s_reagan/reagan_audio_1980s_text_4.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
