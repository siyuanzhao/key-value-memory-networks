"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import metrics
from sklearn.model_selection import train_test_split
from memn2n_kv import MemN2N_KV
from itertools import chain
from six.moves import range, reduce
from memn2n_kv import zero_nil_slot, add_gradient_noise
import time

import tensorflow as tf
import numpy as np
import pandas as pd

timestamp = str(int(time.time()))

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 20, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 40, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("param_output_file", "logs/params_{}.csv".format(timestamp), "Name of output file for model hyperparameters")
tf.flags.DEFINE_string("output_file", "logs/scores_{}.csv".format(timestamp), "Name of output file for final bAbI accuracy scores.")
tf.flags.DEFINE_integer("feature_size", 50, "Feature size")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model")
FLAGS = tf.flags.FLAGS

print("\nParameters:")
with open(FLAGS.param_output_file, 'w') as f:
    for attr, value in sorted(FLAGS.__flags.items()):
        line = "{}={}".format(attr.upper(), value)
        f.write(line + '\n')
        print(line)
    print("")

print("Started Joint Model")

# load all train/test data
ids = range(1, 21)
train, test = [], []
for i in ids:
    tr, te = load_task(FLAGS.data_dir, i)
    train.append(tr)
    test.append(te)
data = list(chain.from_iterable(train + test))

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
trainS = []
valS = []
trainQ = []
valQ = []
trainA = []
valA = []
for task in train:
    S, Q, A = vectorize_data(task, word_idx, sentence_size, memory_size)
    ts, vs, tq, vq, ta, va = train_test_split(S, Q, A, test_size=0.1, random_state=FLAGS.random_state)
    trainS.append(ts)
    trainQ.append(tq)
    trainA.append(ta)
    valS.append(vs)
    valQ.append(vq)
    valA.append(va)

trainS = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainS))
trainQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainQ))
trainA = reduce(lambda a,b : np.vstack((a,b)), (x for x in trainA))
valS = reduce(lambda a,b : np.vstack((a,b)), (x for x in valS))
valQ = reduce(lambda a,b : np.vstack((a,b)), (x for x in valQ))
valA = reduce(lambda a,b : np.vstack((a,b)), (x for x in valA))

testS, testQ, testA = vectorize_data(list(chain.from_iterable(test)), word_idx, sentence_size, memory_size)

n_train = trainS.shape[0]
n_val = valS.shape[0]
n_test = testS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

print(trainS.shape, valS.shape, testS.shape)
print(trainQ.shape, valQ.shape, testQ.shape)
print(trainA.shape, valA.shape, testA.shape)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

# This avoids feeding 1 task after another, instead each batch has a random sampling of tasks
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # decay learning rate
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 90000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

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

    for i in range(1, FLAGS.epochs+1):
        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            train_step(s, q, a)

        if i % FLAGS.evaluation_interval == 0 or i == FLAGS.epochs:
            train_accs = []
            for start in range(0, n_train, n_train/20):
                end = start + n_train/20
                s = trainS[start:end]
                q = trainQ[start:end]
                predict_op = test_step(s, q)
                acc = metrics.accuracy_score(predict_op, train_labels[start:end])
                train_accs.append('{0:.2f}'.format(acc))

            val_accs = []
            for start in range(0, n_val, n_val/20):
                end = start + n_val/20
                s = valS[start:end]
                q = valQ[start:end]
                val_preds = test_step(s, q)
                
                acc = metrics.accuracy_score(np.array(val_preds), val_labels[start:end])

                val_accs.append('{0:.2f}'.format(acc))

            test_accs = []
            for start in range(0, n_test, n_test/20):
                end = start + n_test/20
                s = testS[start:end]
                q = testQ[start:end]

                val_preds = test_step(s, q)
                acc = metrics.accuracy_score(np.array(val_preds), test_labels[start:end])
                test_accs.append('{0:.2f}'.format(acc))

            print('-----------------------')
            print('Epoch', i)
            print()
            t = 1
            for t1, t2, t3 in zip(train_accs, val_accs, test_accs):
                print("Task {}".format(t))
                print("Training Accuracy = {}".format(t1))
                print("Validation Accuracy = {}".format(t2))
                print("Testing Accuracy = {}".format(t3))
                print()
                t += 1
            print('-----------------------')

        # Write final results to csv file
        if i == FLAGS.epochs:
            print('Writing final results to {}'.format(FLAGS.output_file))
            df = pd.DataFrame({
                'Training Accuracy': train_accs,
                'Validation Accuracy': val_accs,
                'Testing Accuracy': test_accs
            }, index=range(1, 21))
            df.index.name = 'Task'
            df.to_csv(FLAGS.output_file)
