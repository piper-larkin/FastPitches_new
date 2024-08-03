#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}

: ${FASTPITCH:="./TC_all/TC_LJ_smaller_out/FastPitch_checkpoint_950.pt"}  # Changed 
: ${BATCH_SIZE:=16}     # changed from 32
: ${PHRASES:="./TC_all/tc_audio_pitch_text_spk_age_phrases_30.tsv"}    # was "phrases/devset10.tsv" or phrases/devset_1994.tsv or phrases/testset_1to30_80s.tsv
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${NUM_SPEAKERS:=18} # changed for LJ

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

# for AGE in 18 45 73 100; do
for AGE in 20 45 85; do
# for AGE in $(seq 80 100); do
    # for SPEAKER in 0 3 15; do
    for SPEAKER in 17; do
        OUTPUT_DIR="./TC_all/TC_LJ_smaller_out/phrases30_${AGE}_${SPEAKER}_950"  # changed dir name
        LOG_FILE="$OUTPUT_DIR/nvlog_infer.json"

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
        [ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 0.0"
        [ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
        [ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

        mkdir -p "$OUTPUT_DIR"

        # Run inference
        python inference.py $ARGS "$@"
    done
done
