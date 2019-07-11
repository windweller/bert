#!/usr/bin/env bash

# extract features from pre-generated SQuAD passages
# we just want the last layer
BERT_BASE_DIR=/mnt/fs5/anie/bert_models/cased_L-12_H-768_A-12
export CUDA_VISIBLE_DEVICES=7

python3.6 extract_features.py \
  --input_file=/mnt/fs5/anie/compress_bert/squad_passages_shorter_than_325.txt \
  --output_file=/mnt/fs5/anie/compress_bert/squad_passages_encoded.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=325 \
  --batch_size=12