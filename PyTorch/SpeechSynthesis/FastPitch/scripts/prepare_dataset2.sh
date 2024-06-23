#!/usr/bin/env bash

set -e

: ${DATA_DIR:=test_LJ_reagan} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/ljs_reagan_audio_text.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS
