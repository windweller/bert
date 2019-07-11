#!/usr/bin/env bash

# model.ckpt-299430
export BERT_BASE_DIR=/mnt/fs5/anie/gs_twitterbert/twitter-bert/char_bert_base_2019_june22_no_2017_04_sent_level_epoch25_spm_bpe_uncased_tpuv2_256
export DATA_DIR=/mnt/fs5/anie/twitter_quote_reply_2019_apr2/holdout/
export CUDA_VISIBLE_DEVICES=4
export TASK_NAME=sep

python3.6 run_validation.py \
  --task_name=sep \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$DATA_DIR \
  --vocab_file=/mnt/fs5/anie/gs_twitterbert/twitter-bert/bert_twitter_char_vocab_w_speical_toks_no_unused.txt \
  --bert_config_file=/home/anie/bert/bert_twitter_char_base/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt-299430 \
  --max_seq_length=300 \
  --eval_batch_size=32 \
  --output_dir=/mnt/fs5/anie/twitter-bert-evals/social_edge_pred/spm_bpe_bert/ \
  --log_file

# model.ckpt-36815
# model.ckpt-88318
# model.ckpt-58879