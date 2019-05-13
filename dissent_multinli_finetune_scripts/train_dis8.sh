#!/usr/bin/env bash

export BERT_BASE_DIR=/home/anie/bert_models/uncased_L-12_H-768_A-12
export DIS_DIR=/mnt/fs5/anie/DisSent-Processed-data
export CUDA_VISIBLE_DEVICES=5

python3.6 run_classifier.py \
  --task_name=dis8 \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DIS_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --save_checkpoints_steps=10000 \
  --num_train_epochs=1.0 \
  --output_dir=/home/anie/bert_models/fine_tuned_dis8_epoch1/ \
  --log_file