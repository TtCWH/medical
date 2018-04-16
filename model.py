# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Model(object):
	def __init__(self, kind_num,vocab_size,seq_length,embedding_dim=300,epochs=100,
		is_training=True,init_lr=0.005,lstm_dim=300,keep_prob=0.5):
		"""
		epochs:训练轮数
		embedding_dim:embedding维度
		seq_length:句子长度
		is_training:是否训练状态
		init_lr:初始learning rate
		lstm_dim:LSTM层维度
		kind_num:kind的个数
		vocab_size:词汇表个数
		"""
		num_steps=seq_length
		self._input_kind=tf.placeholder(tf.int32,[None,142])
		self._input_content=tf.placeholder(tf.int32,[None,142,seq_length])
		self._y_label=tf.placeholder(tf.float32,[None,5])

		#embedding layer
		with tf.device('/cpu:0'),tf.name_scope("embedding_layer"):
			self.kind_embedding=tf.get_variable('kind_embedding',[kind_num,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			self.word_embedding=tf.get_variable('word_embedding',[vocab_size,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			kind_embedding_output=tf.nn.embedding_lookup(self.kind_embedding,self._input_kind)
			word_embedding_output=tf.nn.embedding_lookup(self.word_embedding,self._input_content)
		if is_training and keep_prob<1.0:
			kind_embedding_output=tf.nn.dropout(embedding_output,keep_prob)
			word_embedding_output=tf.nn.dropout(embedding_output,keep_prob)

		#sequence embedding layer
		bilstm_input=tf.unstack(word_embedding_output,num_steps,-2)
		forward_lstm=encoderLSTM(lstm_dim/2,initializer=tf.random_uniform_initializer(-0.01, 0.01),forget_bias=0.0)
		backward_lstm=encoderLSTM(lstm_dim/2,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
		biLSTM_output=tf.contrib.rnn.static_bidirectional_rnn(forward_LSTM,backward_LSTM,self.LSTM_input,dtype=tf.float32)[0]
		self.biLSTM_output=tf.stack(biLSTM_output,axis=-2)
		self.seq_embedding=tf.reduce_mean(self.biLSTM_output,axis=[0,1,3])

