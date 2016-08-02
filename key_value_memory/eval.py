# the code is originally taken from https://github.com/dennybritz/cnn-text-classification-tf
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from memn2n_cnn import MemN2N_KV
import os
import time
import datetime
from nltk.corpus import stopwords
import data_helpers
import sys
import re
import xml.etree.ElementTree as et

# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300,
                        "Dimensionality of character embedding (default: 256)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 200)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

# parameters for MenN2N
tf.flags.DEFINE_integer("hops", 4, "Number of hops in the Memory Network.")

tf.flags.DEFINE_integer(
    "note_words_limit", 400, "Max number of words in each note")

tf.flags.DEFINE_integer(
    "vocab_limit", 10000, "Max number of vocabulary")
tf.flags.DEFINE_integer(
    "feature_size", 300, "The size of feature in memory networks")
tf.flags.DEFINE_string(
    "reader", 'bow',
    'The name of reader for reading medical notes and wiki pages. (gru, bow)')

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data. Load your own data here
# Load data
print("Loading data...")
json_file = "data/pruned_wiki_json_no_stopwords.json"
notes_file = 'data/new_examples_no_stopwords.xml'

r = data_helpers.load_memory_and_notes(
    json_file, notes_file, sys.maxint, sys.maxint, FLAGS.note_words_limit, FLAGS.vocab_limit)
description_list = r[0]
labels_list = r[1]
memory_value = r[2]
memory_key = r[3]
vocabulary = r[4]
max_note_length = r[5]
max_title_sentence_length = r[6]
titles = r[7]

memory_key_size = len(memory_key)
memory_value_size = len(memory_value)
segment_ids = []
memory_key_flat = []
num_doc_words = 0

input_list = []
training_labels = []
# flat labels_list and copy training data
for idx, ite in enumerate(labels_list):
    size = len(ite)
    for i in ite:
        input_list.append(description_list[idx])
        training_labels.append(i)

# calculate each segment count in memory value
# the size of memory key
for idx, ite_dict in enumerate(memory_key):
    num_words = 0
    tmp_page = []
    # k is section name
    for k, v in ite_dict.iteritems():
        for ite in v:
            for i in ite:
                num_words += 1
                tmp_page.append(i)
            # memory_key_size += 1
            # segment_ids.append(idx)
            # memory_key_flat.append(ite)
    if num_words > num_doc_words:
        num_doc_words = num_words
    memory_key_flat.append(tmp_page)

# append to the same length
for ite in memory_key_flat:
    diff = num_doc_words - len(ite)
    if diff > 0:
        for i in range(diff):
            ite.append(0)

vocab_size = FLAGS.vocab_limit+1

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    memn2n = MemN2N_KV(
        batch_size=FLAGS.batch_size,
        vocab_size=vocab_size,
        note_size=max_note_length,
        doc_size=num_doc_words,
        memory_key_size=memory_key_size,
        feature_size=FLAGS.feature_size,
        memory_value_size=memory_value_size,
        embedding_size=FLAGS.embedding_dim,
        reader=FLAGS.reader,
        hops=FLAGS.hops,
        is_training=False
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # restore variables
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        while(True):
            # read note from xml file
            xml = et.parse('topics2016.xml')
            root = xml.getroot()
            children = root.getchildren()
            for child in children:
                note = child[0].text
                input = note
                # remove stop words
                pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
                input = pattern.sub('', input)

                tokenzied_input = []
                wl = data_helpers.clean_str(input)
                count = 0
                for ite in wl:
                    if count < FLAGS.note_words_limit:
                        if ite in vocabulary:
                            tokenzied_input.append(vocabulary[ite])
                        else:
                            tokenzied_input.append(0)
                        count += 1
                    else:
                        break
                print tokenzied_input
                diff = FLAGS.note_words_limit - len(tokenzied_input)
                if(diff > 0):
                    for i in range(diff):
                        tokenzied_input.append(0)

                tokenzied_input = [tokenzied_input]
                feed_dict = {
                    memn2n._doc: memory_key_flat,
                    memn2n._query: tokenzied_input,
                    memn2n._memory_value: memory_value
                }
                probs = sess.run(memn2n.probs, feed_dict)
                print probs
                results = probs[0]
                indices = sorted(range(len(results)), key=lambda i: results[i])[-10:]
                with open('mn_results', 'a') as f:
                    for ite in indices:
                        f.write(titles[ite] + "\n")
                    f.write('\n')

            print 'Job is done. Check the results!'
            exit(0)
