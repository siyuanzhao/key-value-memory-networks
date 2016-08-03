"""Key Value Memory Networks with GRU reader.
The implementation is based on https://arxiv.org/abs/1606.03126
The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from six.moves import range
import numpy as np
# from attention_reader import Attention_Reader

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

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

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

class MemN2N_KV(object):
    """Key Value Memory Network."""
    def __init__(self, batch_size, vocab_size,
                 note_size, doc_size, memory_key_size,
                 memory_value_size, embedding_size,
                 feature_size=30,
                 hops=3,
                 reader='bow',
                 debug_mode=True,
                 is_training=True,
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

        debug_mode: If true, print some debug info about tensors
        name: Name of the End-To-End Memory Network.\
        Defaults to `KeyValueMemN2N`.
        """
        self._is_training = is_training
        self._doc_size = doc_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._note_size = note_size
        self._wiki_sentence_size = doc_size
        self._memory_key_size = memory_key_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._name = name
        self._memory_value_size = memory_value_size
        self._n_hidden = 10
        self._encoding = tf.constant(position_encoding(self._doc_size, self._embedding_size), name="encoding")
        self._build_inputs()

        d = feature_size
        self._feature_size = feature_size
        self._n_hidden = d

        # trainable variables
        if 'gru' == reader:
            self.A = tf.get_variable(
                'A', shape=[d, 2*self._n_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            self.A_mvalue = tf.get_variable(
                'A_mvalue', shape=[d, self._embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())
        elif reader == 'bow':
            self.A = tf.get_variable(
                'A', shape=[d, self._embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            self.A_mvalue = self.A
        elif reader == 'simple_gru':
            self.A = tf.get_variable(
                'A', shape=[d, self._n_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            self.A_mvalue = tf.get_variable(
                'A_mvalue', shape=[d, self._embedding_size],
                initializer=tf.contrib.layers.xavier_initializer())

        # use attention reader to embed notes and wiki pages
        # self.ar = Attention_Reader(
        #    self._batch_size, self._vocab_size, self._embedding_size,
        #    self._note_size, self._memory_value_size, doc_size, d, 3)
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            nil_word_slot = tf.zeros([1, embedding_size])
            self.W = tf.concat(
                0, [nil_word_slot, tf.get_variable(
                    'W', shape=[vocab_size-1, embedding_size],
                    initializer=tf.random_normal_initializer(stddev=0.1))])
            self.W_memory = tf.concat(
                0, [nil_word_slot, tf.get_variable(
                    'W_memory', shape=[vocab_size-1, embedding_size],
                    initializer=tf.random_normal_initializer(stddev=0.1))])
            # self.W_memory = self.W
            self._nil_vars = set([self.W.name, self.W_memory.name])
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self._query)
            # self.embedded_chars_bow = tf.reduce_sum(
            #    self.embedded_chars, 1)
            # word embedding on memory keys (wiki content)
            self.mkeys_embedded_chars = tf.nn.embedding_lookup(
                self.W_memory, self._doc)

            self.mvalues_embedded_chars = tf.nn.embedding_lookup(
                self.W_memory, self._memory_value)
            # [memory size * embedding size]
        if reader == 'gru':
            # read wiki pages
            # Permuting batch_size and n_steps
            x = tf.transpose(self.mkeys_embedded_chars, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self._embedding_size])
            # Split to get a list of 'n_steps'
            # tensors of shape (doc_num, n_input)
            x = tf.split(0, self._doc_size, x)

            # do the same thing on medical notes
            q = tf.transpose(self.embedded_chars, [1, 0, 2])
            q = tf.reshape(q, [-1, self._embedding_size])
            q = tf.split(0, self._note_size, q)

            # feed query to gru
            rnn_fw = tf.nn.rnn_cell.GRUCell(self._n_hidden)
            rnn_bw = tf.nn.rnn_cell.GRUCell(self._n_hidden)
            # get gru cell output
            q_outputs, q_fw, q_bw = tf.nn.bidirectional_rnn(
                rnn_fw, rnn_bw, q, dtype=tf.float32)
            q_r = tf.concat(1, [q_fw, q_bw])

            # feed wiki pages to gru
            d_fw_gru = tf.nn.rnn_cell.GRUCell(self._n_hidden)
            d_bw_gru = tf.nn.rnn_cell.GRUCell(self._n_hidden)

            with tf.variable_scope('doc_gru'):
                _, doc_fw, doc_bw = tf.nn.bidirectional_rnn(d_fw_gru, d_bw_gru, x, dtype=tf.float32)

            doc_r = tf.concat(1, [doc_fw, doc_bw])
        elif reader == 'bow':
            with tf.device('/cpu:0'):
                q_r = tf.reduce_sum(self.embedded_chars*self._encoding, 1)
                doc_r = tf.reduce_sum(self.mkeys_embedded_chars*self._encoding, 2)
                value_r = tf.reduce_sum(self.mvalues_embedded_chars*self._encoding, 2)
        elif reader == 'simple_gru':
            # Permuting batch_size and n_steps
            # for d in ['/gpu:0', '/gpu:1']:
            # with tf.device(d):
            x = tf.transpose(self.mkeys_embedded_chars, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self._embedding_size])
            # Split to get a list of 'n_steps'
            # tensors of shape (doc_num, n_input)
            x = tf.split(0, self._doc_size, x)

            # do the same thing on medical notes
            q = tf.transpose(self.embedded_chars, [1, 0, 2])
            q = tf.reshape(q, [-1, self._embedding_size])
            q = tf.split(0, self._note_size, q)

            rnn = tf.nn.rnn_cell.GRUCell(self._n_hidden)
            q_rnn = tf.nn.rnn_cell.GRUCell(self._n_hidden)

            with tf.variable_scope('gru'):
                doc_output, _ = tf.nn.rnn(rnn, x, dtype=tf.float32)
            with tf.variable_scope('gru', reuse=True):
                q_output, _ = tf.nn.rnn(q_rnn, q, dtype=tf.float32)
            doc_r = doc_output[-1]
            value_r = doc_r
            q_r = q_output[-1]

        r_list = []
        for _ in range(self._hops):
            # define R for variables
            R = tf.get_variable(
                'R{}'.format(_), shape=[self._feature_size, self._feature_size],
                initializer=tf.random_normal_initializer(stddev=0.1))
            r_list.append(R)

        # mkeys is vector representation for wiki titles
        # mvalues is vector representation for wiki pages
        # note is the vector representation for medical notes
        # o = self._key_addressing(
        #    tf.transpose(doc_r, [2, 0, 1]),
        #    tf.transpose(value_r, [2, 0, 1]),
        #    tf.transpose(q_r), r_list)
        # o.shape: [feature_size, batch_size]
        o = self._key_addressing(doc_r, value_r, q_r, r_list)
        o = tf.transpose(o)
        self.B = tf.get_variable(
            'B', shape=[self._feature_size, self._embedding_size],
            initializer=tf.random_normal_initializer(stddev=0.1))

        if debug_mode:
            print "shape of output is ", o.get_shape()
            # print "shape of mkeys_embedded_chars: {}".format(
            #    self.mkeys_embedded_chars.get_shape())
        # q_tmp = tf.matmul(logits, self.B, transpose_a=True)
        # y_tmp.shape: [feature_size, vocab]
        y_tmp = tf.matmul(self.A, self.W_memory, transpose_b=True)
        with tf.name_scope("prediction"):
            # logits = tf.matmul(q_tmp, tf.transpose(self.W))
            logits = tf.matmul(o, y_tmp)
            probs = tf.nn.softmax(tf.cast(logits, tf.float32))
            if debug_mode:
                print 'shape of probs: {}'.format(probs.get_shape())
            if self._is_training:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits, tf.cast(self._labels, tf.float32), name='cross_entropy')
                cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

                # loss op
                loss_op = cross_entropy_sum
                # loss_op = cross_entropy_sum
                # predict ops
                predict_op = tf.argmax(probs, 1, name="predict_op")

                # assign ops
                self.loss_op = loss_op
                self.predict_op = predict_op
            self.probs = probs

    def _build_inputs(self):
        with tf.name_scope("input"):
            
            self._doc = tf.placeholder(
                tf.int32, [None, self._memory_value_size, self._doc_size],
                name='stories')
            self._query = tf.placeholder(
                tf.int32, [None, self._note_size], name='question')

            self._memory_value = tf.placeholder(
                tf.int32, [None, self._memory_value_size, self._doc_size])
            # actual output
            if self._is_training:
                self._labels = tf.placeholder(tf.float32, [None, self._vocab_size], name='answer')

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
            u = tf.matmul(self.A, notes, transpose_b=True)
            u = [u]
            for _ in range(self._hops):
                R = r_list[_]
                u_temp = u[-1]
                # [embedding_size, batch_size x memory_size]
                k_temp = tf.reshape(tf.transpose(mkeys, [2, 0, 1]), [self._embedding_size, -1])
                
                a_k_temp = tf.matmul(self.A, k_temp)
                a_k = tf.reshape(tf.transpose(a_k_temp), [-1, self._memory_key_size, self._feature_size])
                # k_temp = tf.matmul(self.A, tf.reshape(mkeys, [self._embedding_size, -1]))
                u_expanded = tf.expand_dims(tf.transpose(u_temp), [1])
                # reshape k_temp to [batch_size, memory_size, feature_size]
                # k_temp = tf.reshape(k_temp, [-1, self._memory_key_size, self._feature_size])
                # reshape u_temp to [batch_size, 1, feature_size] to do dot product
                # u_temp_reshaped = tf.expand_dims(tf.transpose(u_temp), 1)
                # u_temp_reshaped = tf.transpose(u_temp_expanded, [1, 2, 0])
                # dotted = tf.matmul(u_temp, k_temp, transpose_a=True)
                # do dot product
                # dotted = tf.reduce_sum(k_temp * u_temp_reshaped, 2)
                dotted = tf.reduce_sum(a_k*u_expanded, 2)

                # print 'shape of k_temp: {}'.format(k_temp.get_shape())
                print 'shape of dotted: {}'.format(dotted.get_shape())

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                print 'shape of probs: {}'.format(probs.get_shape())
                probs_expand = tf.expand_dims(probs, -1)
                v_temp = tf.reshape(tf.transpose(mvalues, [2, 0, 1]), [self._embedding_size, -1])
                a_v_temp = tf.matmul(self.A_mvalue, v_temp)
                a_v = tf.reshape(tf.transpose(a_v_temp), [-1, self._memory_key_size, self._feature_size])
                # reshape v_temp to [batch_size, sentence_size, feature_size]
                # v_temp_reshaped = tf.reshape(tf.transpose(v_temp), [-1, self._memory_value_size, self._feature_size])
                # v_temp_expand = tf.expand_dims(v_temp, 0)
                # print 'shape of probs*v_temp: {}'.format(
                #    (probs_expand*v_temp_reshaped).get_shape())
                # o_k = tf.reduce_sum(probs_expand*v_temp_reshaped, 1)
                # [batch_size, feature_size]
                o_k = tf.reduce_sum(probs_expand*a_v, 1)
                o_k = tf.transpose(o_k)
                print 'shape of o_k: {}'.format(o_k.get_shape())

                print 'shape of u[-1]: {}'.format(u[-1].get_shape())
                print 'shape of u[-1]+o_k: {}'.format((u[-1]+o_k).get_shape())
                u_k = tf.matmul(R, u[-1]+o_k)

                u.append(u_k)

            return u[-1]
