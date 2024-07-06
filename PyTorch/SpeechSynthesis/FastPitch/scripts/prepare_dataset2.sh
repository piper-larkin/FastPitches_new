#!/usr/bin/env bash

set -e

: ${DATA_DIR:=reagan_all} # CHANGED
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/reagan_all/reagan_audio_text_age_spk.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    --n-speakers 1 \
    $ARGS
