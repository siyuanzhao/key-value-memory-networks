"""Key Value Memory Networks.
The implementation is based on https://arxiv.org/abs/1606.03126
The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from six.moves import range


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MemN2N_KV(object):
    """Key Value Memory Network."""
    def __init__(self, batch_size, vocab_size,
                 note_size, wiki_sentence_size, memory_key_size,
                 memory_value_size, embedding_size,
                 hops=3,
                 max_grad_norm=40.0,
                 debug_mode=True,
                 name='KeyValueMemN2N'):
        """Creates an Key Value Memory Network

        Args:
        batch_size: The size of the batch.

        vocab_size: The size of the vocabulary
        (should include the nil word). The nil word
            one-hot encoding should be 0.
        note_size: longest number of sentences in medical notes

        wiki_sentence_size: longest number of sentences in wiki pages

        embedding_size: The size of the word embedding.

        memory_key_size: the size of memory slots for keys
        memory_value_size: the size of memory slots for values

        hops: The number of hops. A hop consists of reading and\
        addressing a memory slot.

        max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.
        debug_mode: If true, print some debug info about tensors
        name: Name of the End-To-End Memory Network.\
        Defaults to `KeyValueMemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._note_size = note_size
        self._wiki_sentence_size = wiki_sentence_size
        self._memory_key_size = memory_key_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._name = name
        self._memory_value_size = memory_value_size

        self._build_inputs()

        d = 5

        # trainable variables
        self.A = tf.Variable(
            tf.random_uniform([d, self._embedding_size], -1.0, 1.0))
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            nil_word_slot = tf.zeros([1, embedding_size])
            self.W = tf.concat(
                0, [nil_word_slot, tf.Variable(tf.random_uniform(
                    [vocab_size-1, embedding_size], -1.0, 1.0))])
            #  self.W = tf.Variable(
            #    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self._notes)

            # word embedding on memory keys (wiki titles)
            self.mkeys_embedded_chars = tf.nn.embedding_lookup(
                self.W, self._memory_key)
            self.mkeys_embedded_bow = tf.reduce_sum(
                self.mkeys_embedded_chars, 1)
            self.mkeys_embedded_segment = tf.segment_sum(
                self.mkeys_embedded_bow, self._segment_ids)

            # use bag of words to embed the sentence
            self.mvalues_embedded_chars = tf.nn.embedding_lookup(
                self.W, self._memory_value)
            # [memory size * embedding size]

            self._labels = tf.reshape(self._labels, [-1, 2])
            self.labels = tf.sparse_to_dense(
                self._labels, tf.pack(
                    [self._batch_size, self._memory_value_size]), 1.0,
                name='labels')

        # for the time being, use bag of words to embed medical note

        self.notes_pool_flat = tf.reduce_sum(self.embedded_chars, 1)
        if debug_mode:
            print "shape of notes_pool_flat: {}".format(
                self.notes_pool_flat.get_shape())
        r_list = []
        for _ in range(self._hops):
            # define R for variables
            R = tf.Variable(
                tf.random_uniform([d, d], -1.0, 1.0), name="R")
            r_list.append(R)

        # mkeys is vector representation for wiki titles
        # mvalues is vector representation for wiki pages
        # note is the vector representation for medical notes
        logits = self._key_addressing(
            tf.transpose(self.mkeys_embedded_segment),
            tf.transpose(self.mvalues_embedded_chars),
            tf.transpose(self.notes_pool_flat), r_list)

        self.B = tf.Variable(
            tf.random_uniform([d, self._embedding_size], -1.0, 1.0))
        if debug_mode:
            print "shape of logits is ", logits.get_shape()
            print "shape of mkeys_embedded_chars: {}".format(
                self.mkeys_embedded_chars.get_shape())
        q_tmp = tf.matmul(logits, self.B, transpose_a=True)
        with tf.name_scope("prediction"):
            logits = tf.matmul(
                q_tmp, tf.transpose(self.mvalues_embedded_chars))
            probs = tf.nn.sigmoid(logits)
            if debug_mode:
                print 'shape of probs: {}'.format(probs.get_shape())

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits, self.labels, name='cross_entropy')
            cross_entropy_sum = tf.reduce_sum(
                cross_entropy, name="cross_entropy_sum")

            # loss op
            loss_op = cross_entropy_sum

            # predict ops
            predict_op = tf.argmax(probs, 1, name="predict_op")
            # tweak predict
            indices = tf.range(self._batch_size)*self._memory_key_size
            aligned_predict = predict_op + tf.cast(indices, tf.int64)
            labels_flat = tf.reshape(self.labels, [-1])
            self.gathered_result = tf.gather(labels_flat, aligned_predict)
            # actual_label = tf.argmax(self.labels, 1, name="actual_label")

            # correct_prediction = tf.equal(predict_op, actual_label)
            self.accuracy = tf.reduce_mean(
                tf.cast(self.gathered_result, tf.float32))
            # assign ops
            self.loss_op = loss_op
            self.predict_op = predict_op
            self.probs = probs

    def _build_inputs(self):
        with tf.name_scope("input"):
            # store values in memory
            self._memory_key = tf.placeholder(
                tf.int32, [self._memory_key_size, self._wiki_sentence_size],
                name="memory_keys")
            # store medical notes
            self._notes = tf.placeholder(
                tf.int32, [self._batch_size, self._note_size], name="notes")
            # store keys in memory
            self._memory_value = tf.placeholder(
                tf.int32, [self._memory_value_size])
            # label which wiki page the setences come from
            self._segment_ids = tf.placeholder(
                tf.int32, [self._memory_key_size])
            # actual output
            self._labels = tf.placeholder(
                tf.int32, [None, 2], name='labels')

    '''
    mkeys: the vector representation for keys in memory
    -- shape of each mkeys: [D, 1]
    mvalues: the vector representation for values in memory
    -- shape of each mvalues: [D, 1]
    notes: the vector representation for notes for data set
    -- shape of notes: [D, 1]
    -- shape of R: [d, d]
    -- shape of self.A: [d, D]
    -- shape of self.B: [d, D]
    self.A, self.B and R are the parameters to learn
    '''
    def _key_addressing(self, mkeys, mvalues, notes, r_list):
        with tf.variable_scope(self._name):
            # [d, batch_size]
            u = [tf.matmul(self.A, notes)]
            for _ in range(self._hops):
                R = r_list[_]
                u_temp = u[-1]
                # [d, batch_size]
                k_temp = tf.matmul(self.A, mkeys)
                # k_temp_expanded = tf.transpose(
                #    tf.expand_dims(k_temp, -1), [1, 2, 0])
                # u_temp_expanded = tf.transpose(
                #    tf.expand_dims(u_temp, -1), [1, 2, 0])
                dotted = tf.matmul(u_temp, k_temp, transpose_a=True)

                print 'shape of k_temp: {}'.format(k_temp.get_shape())
                print 'shape of dotted: {}'.format(dotted.get_shape())

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                print 'shape of probs: {}'.format(probs.get_shape())
                probs_expand = tf.expand_dims(probs, 1)
                v_temp = tf.matmul(self.A, mvalues)
                # value reading
                v_temp_expand = tf.expand_dims(v_temp, 0)
                print 'shape of probs*v_temp: {}'.format(
                    (probs_expand*v_temp_expand).get_shape())
                o_k = tf.reduce_sum(probs_expand*v_temp_expand, 2)
                o_k = tf.transpose(o_k)
                print 'shape of o_k: {}'.format(o_k.get_shape())

                print 'shape of u[-1]: {}'.format(u[-1].get_shape())
                print 'shape of u[-1]+o_k: {}'.format((u[-1]+o_k).get_shape())
                u_k = tf.matmul(R, u[-1] + o_k)

                u.append(u_k)

            return u[-1]
