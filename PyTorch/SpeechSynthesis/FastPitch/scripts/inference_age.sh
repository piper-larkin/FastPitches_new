#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}

: ${FASTPITCH:="./reagan_all/output_2/FastPitch_checkpoint_1000.pt"}  # Changed 
: ${BATCH_SIZE:=32}  
: ${PHRASES:="phrases/testset_1to30_80s.tsv"}    # was "phrases/devset10.tsv" or phrases/devset_1994.tsv or phrases/testset_1to30_80s.tsv
: ${OUTPUT_DIR:="./reagan_all/audio_$(basename ${PHRASES}_1000 .tsv)"}      # changed dir name
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${AGE:=65}
: ${SPEAKER:=1}
: ${NUM_SPEAKERS:=1}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --age $AGE"
ARGS+=" --n-speakers $NUM_SPEAKERS"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@" 
