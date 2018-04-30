# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Model(object):
	def __init__(self, kind_size,vocab_size,seq_length,batchsize=50,embedding_dim=300,epochs=100,
		is_training=True,init_lr=0.005,lstm_dim=300,keep_prob=0.5):
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
		num_steps=seq_length
		self._input_kind=tf.placeholder(tf.int32,[batchsize,kind_size])
		self._input_content=tf.placeholder(tf.int32,[batchsize,kind_size,seq_length])
		self._y_label=tf.placeholder(tf.float32,[batchsize,5])

		#embedding layer
		with tf.device('/cpu:0'),tf.name_scope("embedding_layer"):
			self.kind_embedding=tf.get_variable('kind_embedding',[kind_size,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			self.word_embedding=tf.get_variable('word_embedding',[vocab_size,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			kind_embedding_output=tf.nn.embedding_lookup(self.kind_embedding,self._input_kind)
			word_embedding_output=tf.nn.embedding_lookup(self.word_embedding,self._input_content)
		#pdb.set_trace()
		if is_training and keep_prob<1.0:
			kind_embedding_output=tf.nn.dropout(kind_embedding_output,keep_prob)
			word_embedding_output=tf.nn.dropout(word_embedding_output,keep_prob)
		kind_embedding_output=tf.unstack(kind_embedding_output,axis=0)

		#sequence embedding layer
		"""output_tensor=[batchsize,kind_size,seq_length,300]"""
		with tf.variable_scope('biLSTM'):
			# pdb.set_trace()
			bilstm_input=tf.unstack(word_embedding_output,axis=1)
			forward_LSTM=LSTMCell(lstm_dim/2,initializer=tf.random_uniform_initializer(-0.01, 0.01),forget_bias=0.0)
			backward_LSTM=LSTMCell(lstm_dim/2,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
			biLSTM_output=[]
			for k in bilstm_input:
				k=tf.unstack(k,num_steps,-2)
				temp=tf.contrib.rnn.static_bidirectional_rnn(forward_LSTM,backward_LSTM,k,dtype=tf.float32)[0]
				temp=tf.stack(temp,axis=-2)
				biLSTM_output.append(temp)
				# pdb.set_trace()
			# pdb.set_trace()
			biLSTM_output=tf.stack(biLSTM_output,axis=1)
			biLSTM_output=tf.unstack(biLSTM_output,axis=0)

		#attention layer
		"""output_tensor=[batchsize,kind_size,300]"""
		with tf.variable_scope('Attention'):
			attention_M=tf.get_variable('attention_matrix',[embedding_dim,embedding_dim],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=tf.float32)
			a=[]
			# pdb.set_trace()
			for per1,per2 in zip(kind_embedding_output,biLSTM_output):
				per1=tf.unstack(per1,axis=0) #[300]
				per2=tf.unstack(per2,axis=0) #[seq_length,300]
				temp=[]
				for kind,descrip in zip(per1,per2):
					# pdb.set_trace()
					score=tf.matmul(descrip,attention_M)
					score=tf.nn.softmax(tf.matmul(score,tf.reshape(kind,shape=[embedding_dim,1])))
					temp.append(tf.reshape(tf.matmul(score,descrip,transpose_a=True),shape=[embedding_dim]))
				a.append(tf.stack(temp,axis=0))
			self.attention_output=attention_output=tf.stack(a,axis=0)
			


if __name__=="__main__":
	a=Model(388,50000,25)
	print(a.attention_output.shape)




