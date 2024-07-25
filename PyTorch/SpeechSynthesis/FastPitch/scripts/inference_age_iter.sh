#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}

: ${FASTPITCH:="./TC_all/graphemes_out/FastPitch_checkpoint_1000.pt"}  # Changed 
: ${BATCH_SIZE:=32}  
: ${PHRASES:="phrases/harvard_list10.tsv"}    # was "phrases/devset10.tsv" or phrases/devset_1994.tsv or phrases/testset_1to30_80s.tsv
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${NUM_SPEAKERS:=17} # will have to change

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

for AGE in 18 45 73 100; do
    for SPEAKER in {0..16}; do
        OUTPUT_DIR="./TC_all/graphemes_out_all/audio_${PHRASES}_${AGE}_${SPEAKER}"  # changed dir name
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
