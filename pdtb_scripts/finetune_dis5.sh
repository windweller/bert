#!/usr/bin/env bash

# http://linuxcommand.org/lc3_wss0120.php

if [ $# -eq 0 ]
  then
    echo "Need to supply argument: -t pdtb_im | pdtb_imex"
    exit 1
fi

while [ "$1" != "" ]; do
    case $1 in
        -t | --task )           shift
                                TASK_NAME=$1
                                ;;
        * )                     echo "not a valid option"
                                exit 1
    esac
    shift
done

#export BERT_BASE_DIR=/home/anie/bert_models/fine_tuned_dis5
export BERT_BASE_DIR=/home/anie/bert_models/fine_tuned_dis5_epoch1
export PDTB_DIR=/mnt/fs5/anie/PDTB/
export CUDA_VISIBLE_DEVICES=0

 # model.ckpt-452327

python3.6 run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$PDTB_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt-90465 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/home/anie/bert_models/${TASK_NAME}_dis5_epoch1/ \
  --log_file

# --output_dir=/home/anie/bert_models/${TASK_NAME}_dis5_epoch2/