"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import metrics
from sklearn.model_selection import train_test_split
from memn2n_kv import MemN2N_KV
from itertools import chain
from six.moves import range

import tensorflow as tf
import numpy as np
from memn2n_kv import zero_nil_slot, add_gradient_noise

tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.2, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 50, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("feature_size", 40, "Feature size")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 30, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 20, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("output_file", "single_scores.csv", "Name of output file for final bAbI accuracy scores.")

FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean(map(len, (s for s, _, _ in data))))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=.1)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

batch_size = FLAGS.batch_size
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # decay learning rate
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

    with tf.Session() as sess:
        
        model = MemN2N_KV(batch_size=batch_size, vocab_size=vocab_size,
                          query_size=sentence_size, story_size=sentence_size, memory_key_size=memory_size,
                          feature_size=FLAGS.feature_size, memory_value_size=memory_size,
                          embedding_size=FLAGS.embedding_size, hops=FLAGS.hops, reader=FLAGS.reader,
                          l2_lambda=FLAGS.l2_lambda)
        grads_and_vars = optimizer.compute_gradients(model.loss_op)

        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in model._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        train_op = optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(s, q, a):
            feed_dict = {
                model._memory_value: s,
                model._query: q,
                model._memory_key: s,
                model._labels: a,
                model.keep_prob: FLAGS.keep_prob
            }
            _, step, predict_op = sess.run([train_op, global_step, model.predict_op], feed_dict)
            return predict_op

        def test_step(s, q):
            feed_dict = {
                model._query: q,
                model._memory_key: s,
                model._memory_value: s,
                model.keep_prob: 1
            }
            preds = sess.run(model.predict_op, feed_dict)
            return preds

        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            train_preds = []
            #for start in range(0, n_train, batch_size):
            for start, end in batches:
                #end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                predict_op = train_step(s, q, a)
                train_preds += list(predict_op)
                
            if t % FLAGS.evaluation_interval == 0:
                # test on train dataset
                train_preds = test_step(trainS, trainQ)
                train_acc = metrics.accuracy_score(train_labels, train_preds)
                print('-----------------------')
                print('Epoch', t)
                print('Training Accuracy: {0:.2f}'.format(train_acc))
                print('-----------------------')

                val_preds = test_step(valS, valQ)
                val_acc = metrics.accuracy_score(np.array(val_preds), val_labels)
                print (val_preds)
                print('-----------------------')
                print('Epoch', t)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')
        # test on train dataset
        train_preds = test_step(trainS, trainQ)
        train_acc = metrics.accuracy_score(train_labels, train_preds)
        train_acc = '{0:.2f}'.format(train_acc)
        # eval dataset
        val_preds = test_step(valS, valQ)
        val_acc = metrics.accuracy_score(val_labels, val_preds)
        val_acc = '{0:.2f}'.format(val_acc)
        # testing dataset
        test_preds = test_step(testS, testQ)
        test_acc = metrics.accuracy_score(test_labels, test_preds)
        test_acc = '{0:.2f}'.format(test_acc)
        print("Testing Accuracy: {}".format(test_acc))
        print('Writing final results to {}'.format(FLAGS.output_file))
        with open(FLAGS.output_file, 'a') as f:
            f.write('{}, {}, {}, {}\n'.format(FLAGS.task_id, test_acc, train_acc, val_acc))
