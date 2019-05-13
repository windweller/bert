#!/usr/bin/env bash

export BERT_BASE_DIR=/home/anie/bert_models/uncased_L-12_H-768_A-12
# export SAVED_MODEL_PATH=/home/anie/bert_models/fine_tuned_dis8/model.ckpt-241000
#export SAVED_MODEL_PATH=/home/anie/bert_models/fine_tuned_dis8/model.ckpt-508502
export SAVED_MODEL_PATH=/home/anie/bert_models/fine_tuned_dis8_epoch1/model.ckpt-101700
export DIS_DIR=/mnt/fs5/anie/DisSent-Processed-data
export CUDA_VISIBLE_DEVICES=5

python3.6 run_classifier.py \
  --task_name=dis8 \
  --do_predict=true \
  --data_dir=$DIS_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$SAVED_MODEL_PATH \
  --max_seq_length=128 \
  --output_dir=/home/anie/bert_models/fine_tuned_dis8_epoch1/