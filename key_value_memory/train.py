
# ! /usr/bin/env python

import sys
import tensorflow as tf
import os
import time
import datetime
import data_helpers
from memn2n_cnn import MemN2N_KV
from memn2n_cnn import zero_nil_slot
import numpy as np

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256,
                        "Dimensionality of character embedding (default: 256)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 200)")
tf.flags.DEFINE_integer("num_epochs", 600,
                        "Number of training epochs (default: 600)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps "
                        "(default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000,
                        "Save model after this many steps (default: 1000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

# parameters for MenN2N
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("max_grad_norm", 40.0, "Maximum gradient norm. 40.0")
tf.flags.DEFINE_integer(
    "note_words_limit", 400, "Max number of words in each note")
tf.flags.DEFINE_integer(
    "wiki_sentences_limit", 100, "Max number of sentences in each wiki")
tf.flags.DEFINE_integer(
    "vocab_limit", 10000, "Max number of vocabulary")
tf.flags.DEFINE_integer(
    "feature_size", 200, "The size of feature in memory networks")
tf.flags.DEFINE_string(
    "reader", 'bow',
    'The name of reader for reading medical notes and wiki pages. (gru, bow)')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

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

# Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(description_list)))
# dl_shuffled = description_list[shuffle_indices]
# l_shuffled = labels_list[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
ngz = -1 * FLAGS.batch_size
dl_train, dl_dev = description_list[:ngz], description_list[ngz:]
l_train, l_dev = labels_list[:ngz], labels_list[ngz:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(dl_train), len(dl_dev)))
print "Size of keys in memory:", str(memory_key_size)
print "Size of values in memory:", str(memory_value_size)
print "Max length of note: {}".format(max_note_length)
print "Max length of wiki page: {}".format(num_doc_words)
# +1 because of nil word in vocab
vocab_size = FLAGS.vocab_limit+1


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

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
            hops=FLAGS.hops
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                               "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        # set up optimizer for training
        # decay learning rate
        starter_learning_rate = 0.2
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate, global_step,
            1000, 0.96, staircase=True)
       
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # gradient pipeline

        grads_and_vars = optimizer.compute_gradients(memn2n.loss_op, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars if g is not None]
        # grads_and_vars = [(add_gradient_noise(g), v)
        #                  for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in memn2n._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        train_op = optimizer.apply_gradients(nil_grads_and_vars,
                                             name="train_op",
                                             global_step=global_step)

        # Checkpoint directory.
        # Tensorflow assumes this directory already
        # exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkppoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        print "Finishing initializing all variables"

        def train_step(memory_values, notes, memory_keys, labels):
            """
            A single training step
            """
            feed_dict = {
                memn2n._memory_value: memory_values,
                memn2n._query: notes,
                memn2n._doc: memory_keys,
                # memn2n._segment_ids: segment_ids,
                memn2n._labels: labels
            }
            _, step, accuracy, labels, predict_op = sess.run(
                [train_op, global_step, memn2n.accuracy, memn2n.labels,
                 memn2n.predict_op],
                feed_dict)

            print 'predicted value: {}'.format(predict_op)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))

        def dev_step(memory_values, notes, memory_keys, labels):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                memn2n._memory_value: memory_values,
                memn2n._query: notes,
                memn2n._doc: memory_keys,
                # memn2n._segment_ids: segment_ids,
                memn2n._labels: labels
            }

            step, accuracy = sess.run(
                [global_step, memn2n.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            # if writer:
            #    writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(dl_train, l_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        count = 1

        l_dev_reshaped = []
        for index, ite in enumerate(l_dev):
            for i in ite:
                l_dev_reshaped.append([index, i])

        for batch in batches:
            q_batch, l_batch = zip(*batch)
            # q_batch = [q for q in q_batch]

            l_batch_reshaped = []
            for index, ite in enumerate(l_batch):
                for i in ite:
                    l_batch_reshaped.append([index, i])

            train_step(memory_value,
                       q_batch, memory_key_flat, l_batch_reshaped)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(memory_value,
                         dl_dev, memory_key_flat, l_dev_reshaped)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, 'checkpoint_',
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
