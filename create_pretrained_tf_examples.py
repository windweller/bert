"""
Mostly to load in epochs and convert/save them into TF example files

Following the function `write_instance_to_example_files`

We won't keep any instance in-memory, instead we will use load one, write one strategy
"""

from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
import numpy as np
import collections
import tensorflow as tf
import tokenization
import os

import time
from tqdm import tqdm, trange
import json
from multiprocessing import Pool as ProcessPool

from tokenization import TwitterBasicTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer


class Logger(object):
  def __init__(self, sys_path, start_epoch):
    if start_epoch == 0:
      self.log_file = open(sys_path, 'w')
    else:
      self.log_file = open(sys_path, 'a')

  def info(self, message, print_m=False):
    self.log_file.write('[' + time.asctime() + ']: ' + message + '\n')
    self.log_file.flush()
    if print_m:
      print(message)

  def close(self):
    self.log_file.close()

  def write_all(self, lines):
      self.log_file.write("\n".join(lines) + '\n')


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def write_instances_to_example_file(epoch):

    # We do a lazy-loading strategy so that the memory won't go crazy

    max_predictions_per_seq = args.max_predictions_per_seq
    max_seq_length = args.max_seq_length

    writer = tf.python_io.TFRecordWriter(str(args.output_dir / f"epoch_{epoch}.tfrecord"))
    total_written = 0

    data_file = args.input_dir / f"epoch_{epoch}.json"
    metrics_file = args.input_dir / f"epoch_{epoch}_metrics.json"

    metrics = json.loads(metrics_file.read_text())
    num_samples = metrics['num_training_examples']
    # instances

    messages = []

    with data_file.open() as f:

        for inst_index in trange(num_samples, position=epoch, desc=f"Epoch {epoch}"):

            line = f.readline()

            line = line.strip()
            example = json.loads(line)

            tokens = example["tokens"]
            segment_ids = example["segment_ids"]
            is_random_next = example["is_random_next"]
            masked_lm_positions = example["masked_lm_positions"]
            masked_lm_labels = example["masked_lm_labels"]

            # this is our hard-threshold, anything longer than this, we dump!
            if len(masked_lm_positions) > max_predictions_per_seq:
                continue

            assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

            input_array = np.zeros(max_seq_length, dtype=np.int)
            input_array[:len(input_ids)] = input_ids

            mask_array = np.zeros(max_seq_length, dtype=np.bool)
            mask_array[:len(input_ids)] = 1

            segment_array = np.zeros(max_seq_length, dtype=np.bool)
            segment_array[:len(segment_ids)] = segment_ids

            lm_position_array = np.full(max_predictions_per_seq, dtype=np.int, fill_value=0)
            lm_position_array[:len(masked_lm_positions)] = masked_lm_positions

            lm_label_array = np.full(max_predictions_per_seq, dtype=np.int, fill_value=0)
            lm_label_array[:len(masked_label_ids)] = masked_label_ids

            masked_lm_weights = [1.0] * len(masked_lm_labels)
            lm_weights_array = np.full(max_predictions_per_seq, dtype=np.int, fill_value=0)
            lm_weights_array[:len(masked_lm_labels)] = masked_lm_weights

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(input_array)
            features["input_mask"] = create_int_feature(mask_array)
            features["segment_ids"] = create_int_feature(segment_array)
            features["masked_lm_positions"] = create_int_feature(lm_position_array)
            features["masked_lm_ids"] = create_int_feature(lm_label_array)
            features["masked_lm_weights"] = create_float_feature(lm_weights_array)
            features["next_sentence_labels"] = create_int_feature([is_random_next])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())  # if each TFRecord is too big, we will try the round-robin strategy

            total_written += 1

            if inst_index < 20:
                messages.append(f"*** Example epoch {epoch} ***")
                messages.append("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in example['tokens']]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    values = []
                    if feature.int64_list.value:
                        values = feature.int64_list.value
                    elif feature.float_list.value:
                        values = feature.float_list.value
                    messages.append(
                        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    writer.close()
    #
    return total_written, epoch, messages


# BERT method that we are not using...
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""

  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    # round-robin strategy...each example is equally distributed into X number of files
    # We will NOT do this...
    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  # tf.logging.info("Wrote %d total instances", total_written)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")  # the CHAR config does not lower case anything...
    parser.add_argument("--do_bpe", action="store_true")
    parser.add_argument("--vocab_path", type=str, required=False, default="",
                        help="The location of the char-vocab file")
    parser.add_argument("--num_workers", type=int, default=40, help="how many epoch files to process at once")
    parser.add_argument("--total_epochs", type=int, default=40, help="how many epoch files to process at once")
    parser.add_argument("--max_seq_length", type=int, default=300, help="should be the same as data generation")
    parser.add_argument("--max_predictions_per_seq", type=int, default=45, help="should be the same as data generation")
    parser.add_argument("--output_dir", type=Path, required=True, help="where to output")
    parser.add_argument("--input_dir", type=Path, required=True, help="where is the input")

    args = parser.parse_args()

    if args.do_bpe:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    else:
        tokenizer = TwitterBasicTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(args.output_dir / "log.txt", 0)

    workers = ProcessPool(args.num_workers)

    for tup in workers.imap(write_instances_to_example_file, range(args.total_epochs)):
        total_written, epoch, messages = tup
        logger.write_all(messages)
        logger.info(f"Wrote {total_written} total instances for epoch {epoch}")

    logger.close()
