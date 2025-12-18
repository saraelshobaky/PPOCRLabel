#!/bin/bash
# To run this script:
# $ cd ~/data/capmas/vsworkspace/capmas_projects/PPOCRLabel

# $ bash training_script.sh > full_training.log 2>&1   
# $ nohup bash training_script.sh > ../logs/full_training.log 2>&1 &  # Background job
# $ bash training_script.sh -f  # Just flipping, only put -f means set to true
# $ bash training_script.sh  --batch-size 64 --learning-rate 0001  #Change everythign


# $ nohup bash training_script.sh  --batch-size 64 --learning-rate 0.0001  > ../logs/full_training.log 2>&1 & 
# $ nohup bash training_script.sh -f -b 64 -lr 0.00025  > ../logs/full_training-f1-b64-lr00025_3fonts_80img.log 2>&1 & 


# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Set Default Values ---
# generating training data configuration variables
FLIP_RTL=false          # Default: Don't flip  # if yes, flip text characters LTR
# Yaml file training configuration parameters
NEW_LR=0.0001           # Default: 0.0001      # Reduced/Set for Fine-tuning
BATCH_SIZE=32           # Default: 64          # Reduced for GPU memory

# --- Parse Command Line Arguments ---
# This loop runs as long as there are arguments ($# > 0)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--flip) 
            FLIP_RTL=true 
            ;;
        -b|--batch-size) 
            BATCH_SIZE="$2"
            shift # Skip the value we just used
            ;;
        -lr|--learning-rate) 
            NEW_LR="$2"
            shift # Skip the value
            ;;       
        *) 
            echo "Unknown parameter passed: $1"
            exit 1 
            ;;
    esac
    shift # Move to the next argument key
done

     

# Derived directory paths for convenience
PAREND_DIR="/home/sara/data/capmas/vsworkspace/capmas_projects"
PPOCRLABEL_DIR="$PAREND_DIR/PPOCRLabel"
PADDLEOCR_DIR="$PAREND_DIR/PaddleOCR"
OUTPUT_DIR="$PAREND_DIR/output"
INFERENCE_DIR="$OUTPUT_DIR/inference"
MODEL_DIR="$OUTPUT_DIR/models"
DATASET_TRAIN_DATA="$PAREND_DIR/train_data"
TRAINING_CONF_PATH="$PPOCRLABEL_DIR/training_conf/arabic_PP-OCRv5_mobile_rec.yaml"

echo '*** Starting the training and evaluation script ***'
echo "Configuration: FLIP_RTL=$FLIP_RTL, BATCH_SIZE=$BATCH_SIZE, LEARNING_RATE=$NEW_LR, PAREND_DIR=$PAREND_DIR"

# -----------------------------------------------------------------------------
# Step 1: Conda Environment Activation
# -----------------------------------------------------------------------------
echo '*** 1. Activating Conda Environment ***'
# Note: The 'paddle-label-train-env' must exist for this to work.
# Using 'source' is often necessary to correctly activate conda environments in a script.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate paddle-label-train-env
echo "Conda environment 'paddle-label-train-env' activated."
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'


# -----------------------------------------------------------------------------
# Step 2: Generate Training Data
# -----------------------------------------------------------------------------
echo '*** 2. Generating training data ***'
cd "$PPOCRLABEL_DIR"
python generate_smart_training_data.py
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'


# -----------------------------------------------------------------------------
# Step 3: Split Train/Val/Test Data
# -----------------------------------------------------------------------------
echo '*** 3. Splitting train_val_test data ***'
python gen_ocr_train_val_test.py \
    --trainValTestRatio 7:3:0 \
    --datasetRootPath "train_data/" \
    --recLabelFileName "Label.txt" \
    --flipRTL "$FLIP_RTL"
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'

# -----------------------------------------------------------------------------
# Step 4: Train the Model
# -----------------------------------------------------------------------------
echo '*** 4. Training the model ***'
cd "$PADDLEOCR_DIR"
python tools/train.py \
    -c "$TRAINING_CONF_PATH" \
    -o Train.loader.batch_size_per_card=$BATCH_SIZE \
       Train.sampler.first_bs=$BATCH_SIZE \
       Eval.loader.batch_size_per_card=$BATCH_SIZE \
       Optimizer.lr.learning_rate=$NEW_LR
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'

# -----------------------------------------------------------------------------
# Step 5: Export Training Model
# -----------------------------------------------------------------------------
echo '*** 5. Exporting the trained model to inference format ***'
# The latest.pdparams will be the output of the training in Step 4
python tools/export_model.py \
    -c "$TRAINING_CONF_PATH" \
    -o Global.pretrained_model="$PADDLEOCR_DIR/output/arabic_rec_ppocr_v5/latest.pdparams" \
       Global.save_inference_dir="$MODEL_DIR/models/arabic_rec_ppocr_v5/tuned-flip_$FLIP_RTL-batchsize_$BATCH_SIZE-learnrate_$NEW_LR/" \
       Train.loader.batch_size_per_card=$BATCH_SIZE \
       Train.sampler.first_bs=$BATCH_SIZE \
       Eval.loader.batch_size_per_card=$BATCH_SIZE \
       Optimizer.lr.learning_rate=$NEW_LR
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'


# -----------------------------------------------------------------------------
# Step 6: Validate Accuracies (using the best_accuracy checkpoint)
# -----------------------------------------------------------------------------
echo '*** 6.Validating Accuracies ***'
# Assuming the best_accuracy checkpoint is saved during training in Step 4
python tools/eval.py \
    -c "$TRAINING_CONF_PATH" \
    -o Global.checkpoints=./output/arabic_rec_ppocr_v5/best_accuracy \
       Train.loader.batch_size_per_card=$BATCH_SIZE \
       Train.sampler.first_bs=$BATCH_SIZE \
       Eval.loader.batch_size_per_card=$BATCH_SIZE \
       Optimizer.lr.learning_rate=$NEW_LR
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'


# -----------------------------------------------------------------------------
# Step 7: Generate Predictions and Log Output
# -----------------------------------------------------------------------------
echo '*** 7. Generating Predictions and logging output ***'
# The output of this command is redirected to a log file
LOG_FILE="$INFERENCE_DIR/flip_$FLIP_RTL-batchsize_$BATCH_SIZE-learnrate_$NEW_LR.log"
echo "Logging inference results to $LOG_FILE"
python tools/infer_rec.py \
    -c "$TRAINING_CONF_PATH" \
    -o Global.checkpoints="./output/arabic_rec_ppocr_v5/best_accuracy" \
       Global.infer_img="$DATASET_TRAIN_DATA/rec/val/" \
       Global.save_res_path=./output/rec/predicts_arabic_ppocrv5.txt \
       Train.loader.batch_size_per_card=$BATCH_SIZE \
       Train.sampler.first_bs=$BATCH_SIZE \
       Eval.loader.batch_size_per_card=$BATCH_SIZE \
       Optimizer.lr.learning_rate=$NEW_LR \
    > "$LOG_FILE" 2>&1
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'
echo '-----------------------------------------------------------------------------'

echo '*** Script execution finished successfully ***'