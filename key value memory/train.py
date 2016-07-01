
# ! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from memn2n_cnn import MemN2N_KV
from memn2n_cnn import add_gradient_noise

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 512,
                        "Dimensionality of character embedding (default: 128)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 400,
                        "Number of training epochs (default: 400)")
tf.flags.DEFINE_integer("evaluate_every", 1000,
                        "Evaluate model on dev set after this many steps "
                        "(default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50000,
                        "Save model after this many steps (default: 1000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

# parameters for MenN2N
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("max_grad_norm", 40.0, "Maximum gradient norm. 40.0")

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
json_file = "data/pruned_wiki_clinial_medicine.json"
notes_file = 'data/new_examples_nonempty.xml'

r = data_helpers.load_memory_and_notes(json_file, notes_file, 120, 10000)
description_list = r[0]
labels_list = r[1]
memory_value = r[2]
memory_key = r[3]
vocabulary = r[4]
max_note_length = r[5]
max_title_sentence_length = r[6]

memory_key_size = 0
memory_value_size = len(memory_value)
segment_ids = []
memory_key_flat = []
# calculate each segment count in memory value
# the size of memory key
for idx, ite_dict in enumerate(memory_key):
    for k, v in ite_dict.iteritems():
        for ite in v:
            memory_key_size += 1
            segment_ids.append(idx)
            memory_key_flat.append(ite)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(description_list)))
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
# +1 because of nil word in vocab
vocab_size = len(vocabulary)+1


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
            wiki_sentence_size=max_title_sentence_length,
            memory_key_size=memory_key_size,
            memory_value_size=memory_value_size,
            embedding_size=FLAGS.embedding_dim,
            hops=FLAGS.hops
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                               "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        # set up optimizer for training
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        # gradient pipeline
        grads_and_vars = optimizer.compute_gradients(memn2n.loss_op)

        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v)
                          for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
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

        def train_step(memory_values, notes, memory_keys, segment_ids, labels):
            """
            A single training step
            """
            feed_dict = {
                memn2n._memory_value: memory_values,
                memn2n._notes: notes,
                memn2n._memory_key: memory_keys,
                memn2n._segment_ids: segment_ids,
                memn2n._labels: labels
            }
            _, step, accuracy, labels, probs = sess.run(
                [train_op, global_step, memn2n.accuracy, memn2n.labels,
                 memn2n.probs],
                feed_dict)
            # print 'actual labels: {}'.format(labels)
            # print 'predicted value: {}'.format(probs)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))

        def dev_step(memory_values, notes, memory_keys, segment_ids, labels):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                memn2n._memory_value: memory_values,
                memn2n._notes: notes,
                memn2n._memory_key: memory_keys,
                memn2n._segment_ids: segment_ids,
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
            q_batch = [q for q in q_batch]

            l_batch_reshaped = []
            for index, ite in enumerate(l_batch):
                for i in ite:
                    l_batch_reshaped.append([index, i])

            train_step(memory_value,
                       q_batch, memory_key_flat, segment_ids, l_batch_reshaped)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(memory_value,
                         dl_dev, memory_key_flat, segment_ids, l_dev_reshaped)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, 'checkpoint_',
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
