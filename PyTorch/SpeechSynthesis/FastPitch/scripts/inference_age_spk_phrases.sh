#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="./TC_all/graphemes_out/FastPitch_checkpoint_1000.pt"}   # changed
: ${BATCH_SIZE:=32}
: ${OUTPUT_DIR:="./TC_all/graphemes_out_reagan_phrases/"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${NUM_SPEAKERS:=17}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

# Define the directory containing your inference files
INFERENCE_DIR="./phrases/reagan_phrase_age"

for file in "$INFERENCE_DIR"/*; do
    # Extract the base file name without the path
    filename=$(basename -- "$file")
    
    # Extract speaker and age from the file name assuming format: tc_audio_pitch_text_spk_age_phrases_3_72.txt
    # speaker=$(echo "$filename" | awk -F'_' '{print $8}')
    # speaker=$(echo "$filename" | awk -F'_' '{print $3}')
    speaker=0
    # age=$(echo "$filename" | awk -F'_' '{print $9}' | awk -F'.' '{print $1}')
    age=$(echo "$filename" | awk -F'_' '{print $4}' | awk -F'.' '{print $1}')


     # Verify extracted values
    echo "Filename: $filename"
    echo "Speaker: $speaker"
    echo "Age: $age"    

    # Set PHRASES to the current file
    PHRASES="$file"

    # Construct output directory and log file based on speaker and age
    # OUTPUT_DIR="${OUTPUT_DIR_BASE}_${speaker}_${age}"
    # LOG_FILE="${LOG_FILE_BASE}_${speaker}_${age}.json"

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
    ARGS+=" --speaker $speaker"
    ARGS+=" --age $age"
    ARGS+=" --n-speakers $NUM_SPEAKERS"
    [ "$CPU" = false ]          && ARGS+=" --cuda"
    [ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
    [ "$AMP" = true ]           && ARGS+=" --amp"
    [ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 0.0"
    [ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
    [ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

    mkdir -p "$OUTPUT_DIR"

    # CUDA_LAUNCH_BLOCKING=1 
    python inference.py $ARGS "$@" # ADDED for debugging
done
