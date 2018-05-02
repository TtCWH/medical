# -*- encoding:utf-8 -*-
__author__ = 'Han Wang/Xuan Hua'
import os
import pdb
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell,GRUCell
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Model(object):
    def __init__(self, kind_size,
                 vocab_size,
                 seq_length,
                 batchsize=50,
                 embedding_dim=300,
                 epochs=100,
                 is_training=True,
                 init_lr=0.005,
                 lstm_dim=300,
                 keep_prob=0.5,
                 cnn_fc_keep_prob=0.6):
        """
        epochs:训练轮数
        embedding_dim:embedding维度
        seq_length:句子长度
        is_training:是否训练状态
        init_lr:初始learning rate
        lstm_dim:LSTM层维度
        kind_size:kind的个数
        vocab_size:词汇表个数
        """
        self.epochs = epochs
        self.is_training = is_training
        self.init_lr = init_lr

        num_steps = seq_length
        self.input_kind = tf.placeholder(tf.int32, [None, kind_size])
        self.input_content = tf.placeholder(tf.int32, [None, kind_size, seq_length])
        self.y_label = tf.placeholder(tf.float32, [None, 5])

        # embedding layer
        print("embedding")
        with tf.device('/cpu:0'), tf.name_scope("embedding_layer"):
            self.kind_embedding = tf.get_variable('kind_embedding', [kind_size, embedding_dim],
                                                  initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                  dtype=tf.float32)
            self.word_embedding = tf.get_variable('word_embedding', [vocab_size, embedding_dim],
                                                  initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                  dtype=tf.float32)
            kind_embedding_output = tf.nn.embedding_lookup(self.kind_embedding, self.input_kind)
            word_embedding_output = tf.nn.embedding_lookup(self.word_embedding, self.input_content)
        # pdb.set_trace()
        if is_training and keep_prob < 1.0:
            kind_embedding_output = tf.nn.dropout(kind_embedding_output, keep_prob)
            word_embedding_output = tf.nn.dropout(word_embedding_output, keep_prob)
        kind_embedding_output=tf.layers.batch_normalization(kind_embedding_output, training=is_training)
        word_embedding_output=tf.layers.batch_normalization(word_embedding_output, training=is_training)
        #kind_embedding_output = tf.unstack(kind_embedding_output, axis=0)

        # sequence embedding layer
        """output_tensor=[batchsize,kind_size,seq_length,300]"""
        print("LSTM")
        with tf.variable_scope('biLSTM'):
            # pdb.set_trace()
            word_embedding_output=tf.reshape(word_embedding_output,shape=[-1,word_embedding_output.shape[-2],word_embedding_output.shape[-1]])
            bilstm_input = tf.unstack(word_embedding_output, axis=1)
            forward_LSTM = GRUCell(lstm_dim, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                    bias_initializer=tf.zeros_initializer())
            backward_LSTM = GRUCell(lstm_dim, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                    bias_initializer=tf.zeros_initializer())
            biLSTM_output = tf.contrib.rnn.static_bidirectional_rnn(forward_LSTM, backward_LSTM, bilstm_input, dtype=tf.float32)[0]
            biLSTM_output = tf.stack(biLSTM_output, axis=1)
            biLSTM_output=tf.reshape(biLSTM_output,shape=[-1,kind_size,seq_length,lstm_dim*2])
            #biLSTM_output = tf.unstack(biLSTM_output, axis=0)

        # attention layer
        """output_tensor=[batchsize,kind_size,300]"""
        print("attention")
        with tf.variable_scope('Attention'):
            attention_M1 = tf.get_variable('attention_matrix1', [embedding_dim, lstm_dim*2],
                                          initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
            attention_M2 = tf.get_variable('attention_matrix2', [1, seq_length],
                                          initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
            kind_embedding_output=tf.reshape(kind_embedding_output,shape=[-1,embedding_dim])
            temp=tf.matmul(kind_embedding_output,attention_M1)
            temp=tf.reshape(tf.expand_dims(temp,axis=-1),shape=[-1,1])
            attention=tf.nn.relu(tf.reshape(tf.matmul(temp,attention_M2),shape=[-1,kind_size,seq_length,lstm_dim*2])*biLSTM_output)
            attention_output = attention*biLSTM_output
            self.attention_output = attention_output = tf.layers.batch_normalization(tf.reduce_mean(attention_output,axis=-2), training=is_training)
#             a = []
#             # pdb.set_trace()
#             for per1, per2 in zip(kind_embedding_output, biLSTM_output):
#                 per1 = tf.unstack(per1, axis=0)  # [300]
#                 per2 = tf.unstack(per2, axis=0)  # [seq_length,300]
#                 temp = []
#                 for kind, descrip in zip(per1, per2):
#                     # pdb.set_trace()
#                     score = tf.matmul(descrip, attention_M)
#                     score = tf.nn.softmax(tf.matmul(score, tf.reshape(kind, shape=[embedding_dim, 1])))
#                     temp.append(tf.reshape(tf.matmul(score, descrip, transpose_a=True), shape=[embedding_dim]))
#                 a.append(tf.stack(temp, axis=0))
#             self.attention_output = attention_output = tf.stack(a, axis=0)

        '''特征提取 卷积'''
        print("CNN")
        with tf.variable_scope('conv1'):
            kernel1 = tf.get_variable('kernel1',[5, lstm_dim*2, 64],initializer=tf.glorot_normal_initializer(), dtype=tf.float32)
            bias1 = tf.get_variable('kernel2',[64],initializer=tf.zeros_initializer(), dtype=tf.float32)
            conv1 = tf.nn.relu(tf.nn.conv1d(self.attention_output, kernel1, 1, 'VALID') + bias1)

        with tf.variable_scope('maxpool1'):
            maxpool1 = self.__max_pool(conv1)

#        with tf.variable_scope('conv2'):
#            kernel2 = tf.Variable(tf.glorot_normal_initializer()([5, 64, 128]))
#            bias2 = tf.Variable(tf.zeros_initializer()([128]))
#            conv2 = tf.nn.relu(tf.nn.conv1d(maxpool1, kernel2, 1, 'VALID') + bias2)

        #with tf.variable_scope('maxpool2'):
        #    maxpool2 = self.__max_pool(conv2)

        flatten = tf.reshape(maxpool1, [-1, (kind_size - 4) // 2 * 64])

        '''fully connection'''
        with tf.variable_scope('fc1'):
            weight1 = tf.get_variable('weight1',[(kind_size - 4) // 2 * 64, 512],initializer=tf.glorot_normal_initializer(), dtype=tf.float32)
            bias6 = tf.get_variable('bias6',[512],initializer=tf.zeros_initializer(), dtype=tf.float32)
            fc1 = tf.nn.relu(tf.matmul(flatten, weight1) + bias6)

        dropout1 = tf.nn.dropout(fc1, cnn_fc_keep_prob)

 #       with tf.variable_scope('fc2'):
 #           weight2 = tf.Variable(tf.glorot_normal_initializer()([2048, 512]))
 #           bias7 = tf.Variable(tf.zeros_initializer()([512]))
 #           fc2 = tf.nn.relu(tf.matmul(dropout1, weight2) + bias7)
#
 #       dropout2 = tf.nn.dropout(fc2, cnn_fc_keep_prob)

        with tf.variable_scope('fc2'):
            weight2 = tf.get_variable('weight2',[512, 5],initializer=tf.glorot_normal_initializer(), dtype=tf.float32)
            bias8 = tf.get_variable('bias8',[5],initializer=tf.zeros_initializer(), dtype=tf.float32)

        self._result = tf.nn.relu(tf.matmul(dropout1, weight2) + bias8)

        if not is_training:
            return

        self._loss = loss = tf.reduce_mean(
            tf.reduce_mean(
                tf.square(
                    tf.log(self._result + 1) - tf.log(tf.clip_by_value(self.y_label,0,200) + 1)
                ), axis=-1))


        global_step=tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_lr, global_step, epochs*100, 0.98, staircase=True)

        optimizer=tf.train.RMSPropOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._train_op=optimizer.minimize(loss)


    '''池化'''
    def __max_pool(self, tensor):
        tensor = tf.expand_dims(tensor, -1)
        maxpool = tf.nn.max_pool(tensor, [1, 2, 1, 1], [1, 2, 1, 1], 'VALID')
        return tf.squeeze(maxpool, -1)

    @property
    def result(self):
        return self._result

    @property
    def loss(self):
        return self._loss

    @property
    def train_step(self):
        return self._train_op

    def set_optimizer(self, optimizer=tf.train.AdamOptimizer, lr=None):
        if lr is None:
            self.optimizer = optimizer(self.init_lr).minimize(self.loss)
        else:
            self.optimizer = optimizer(lr).minimize(self.loss)

    def training_step(self, session, feed_dict, return_loss=True):
        a, b = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
        if return_loss:
            return b

    def validation_step(self, session, feed_dict):
        return session.run(self.loss, feed_dict=feed_dict)


if __name__ == "__main__":
    model = Model(388, 50000, 25)
    input_kind = model.input_kind
    input_content = model.input_content

    y_label = model.y_label

    model.get_loss()
    model.set_optimizer()




