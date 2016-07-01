'''
The implementation of Gated-Attention Reader:
https://arxiv.org/abs/1606.01549
'''
import tensorflow as tf
import numpy as np


class Attention_Reader(object):
    """ Attention Reader for Documents """
    def __init__(self, batch_size, vocab_size, embedding_size, doc_num,
                 doc_size, query_size, n_hidden, k,
                 keep_prob=0.6, is_training=True,
                 name='AttentionReader'):
        """ Creates Attention Reader for Documents
        Args:
        batch_size: the size of batch
        vocab_size: The size of the vocabulary
        (should include the nil word). The nil word
            one-hot encoding should be 0.
        doc_num: total number of documents
        embedding_size: The size of the word embedding.
        doc_size: the size of the document
        k: the number of layers
        """
        self._doc_size = doc_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._query_size = query_size
        self._n_hidden = n_hidden
        self._doc_num = doc_num
        self._k = k
        self._keep_prob = keep_prob
        # build inputs for model
        with tf.name_scope('input'):
            self._doc = tf.placeholder(
                tf.int32, [self._doc_num, self._doc_size], name='doc')
            self._query = tf.placeholder(
                tf.int32, [self._batch_size, self._query_size], name='query')

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            nil_word_slot = tf.zeros([1, embedding_size])
            self.W = tf.concat(
                0, [nil_word_slot, tf.Variable(tf.random_uniform(
                    [vocab_size-1, embedding_size], -1.0, 1.0))])

            word_embeddings = tf.nn.embedding_lookup(self.W, self._doc)
            # Current data input shape: (batch_size, n_steps, embedding)
            # Permuting batch_size and n_steps
            x = tf.transpose(word_embeddings, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self._embedding_size])
            # Split to get a list of 'n_steps'
            # tensors of shape (doc_num, n_input)
            x = tf.split(0, self._doc_size, x)

            q_we = tf.nn.embedding_lookup(self.W, self._query)
            q = tf.transpose(q_we, [1, 0, 2])
            q = tf.reshape(q, [-1, self._embedding_size])
            q = tf.split(0, self._query_size, q)

        sl = None
        for i in range(self._k):
            with tf.name_scope(
                    'bidirectional_gru'), tf.variable_scope(
                        'query_gru_{}_layer'.format(i)):
                rnn_fw = tf.nn.rnn_cell.GRUCell(self._n_hidden)
                rnn_bw = tf.nn.rnn_cell.GRUCell(self._n_hidden)
                # get gru cell output
                q_outputs, q_fw, q_bw = tf.nn.bidirectional_rnn(
                    rnn_fw, rnn_bw, q, dtype=tf.float32)
                q_r = tf.concat(1, [q_fw, q_bw])

                d_fw_gru = tf.nn.rnn_cell.GRUCell(self._n_hidden)
                d_bw_gru = tf.nn.rnn_cell.GRUCell(self._n_hidden)

            if i == 0:
                with tf.variable_scope('doc_gru'):
                    doc_outputs, _, __ = tf.nn.bidirectional_rnn(
                        d_fw_gru, d_bw_gru, x, dtype=tf.float32)
            else:
                vs_str = 'doc_gru_{}_layer'.format(i)
                # define another bidirectional_rnn for hidden layers
                with tf.variable_scope(vs_str):

                    doc_outputs, _, __ = tf.nn.bidirectional_rnn(
                        d_fw_gru, d_bw_gru, sl, dtype=tf.float32)
            if i == self._k-1:
                final_state = tf.reduce_sum(tf.mul(doc_outputs, q_r), 2)
                # shape: [doc_num, doc_size]
                final_state = tf.transpose(final_state)
            else:
                state = tf.mul(doc_outputs, q_r)
                if is_training:
                    state = tf.nn.dropout(state, self._keep_prob)
                # reshape to a list
                sl = tf.reshape(state, [-1, 2*self._n_hidden])
                sl = tf.split(0, self._doc_size, sl)
        # shape: [doc_num, doc_size]
        probs = tf.nn.softmax(final_state)
        # [doc_size, doc_num, 2*hidden_size] -->
        # [doc_num, doc_size, 2*hidden_size]
        t_doc_outputs = tf.transpose(doc_outputs, [1, 0, 2])
        # [doc_num, doc_size, 1]
        probs_e = tf.expand_dims(probs, -1)
        doc_r = tf.reduce_sum(tf.mul(t_doc_outputs, probs_e), 1)

        # assign op
        self.doc_outputs = t_doc_outputs
        self.probs = probs_e
        self.doc_r = doc_r


def main():
    batch_size = 2
    vocab_size = 30
    embedding_size = 6
    doc_size = 8
    n_hidden = 5
    query_size = 2

    doc_num = 2
    k = 3
    ar = Attention_Reader(
        batch_size, vocab_size, embedding_size, doc_num, doc_size,
        query_size, n_hidden, k)

    # Initializing the variables
    init = tf.initialize_all_variables()
    doc = np.array(
        [i for i in range(doc_num*doc_size)]).reshape([doc_num, doc_size])
    q = np.array(
        [i for i in range(batch_size*query_size)]).reshape(
            [batch_size, query_size])

    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {ar._doc: doc, ar._query: q}
        foutputs = sess.run(ar.doc_r, feed_dict)
        print foutputs
        print 10*'*'
        doc_outputs = sess.run(ar.doc_outputs, feed_dict)
        probs = sess.run(ar.probs, feed_dict)
        print doc_outputs
        print 10*'*'
        print probs

if __name__ == "__main__":
    main()
