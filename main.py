# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import os
import pdb
import time
import numpy as np
import tensorflow as tf
from preprocess import get_data,store_data,Preprocess,get_test_data
from model import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_train_batch(input_kind,input_descrip,inputY,batchsize,shuffle=True):
	"""
	inputE:实体对
	inputR:实体对对应的relation
	inputY:三元组的score
	batchsize:batch大小
	shuffle:打乱数据集
	"""
	assert len(input_kind) == len(input_descrip)
	assert len(input_descrip) == len(inputY)
	indices=np.arange(len(inputY))
	if shuffle:
		np.random.shuffle(indices)
	for start_index in range(0,len(inputY)-batchsize+1,batchsize):
		sub_list=indices[start_index:start_index+batchsize]
		kind=np.zeros((batchsize,387)).astype('int32')
		desc=np.zeros((batchsize,387,500)).astype('int32')
		y=np.zeros((batchsize,5)).astype('float32')
		for i,index in enumerate(sub_list):
			kind[i,]=inputE[index]
			desc[i,]=inputR[index]
			y[i]=inputY[index]
		yield kind,desc,y


def write_log(val,epoch):
	with open("test_epoch{}.log".format(epoch),'a') as f:
		f.write("{}\n".format(val))



def train_model(seq_length,kind_size,vocab_size,epochs=100,batchsize=5,recover=False,current_epoch=0):
	"""
	epochs:训练轮数
	"""
	#kind_test,desc_test=get_test_data()
	#kind_test=np.asarray(kind_test,dtype="int32")
	#desc_test=np.asarray(desc_test,dtype="int32")

	with tf.Session() as sess:
		with tf.variable_scope("model",reuse=None):
			m=Model(kind_size,vocab_size,seq_length,batchsize=batchsize,init_lr=0.001)
		with tf.variable_scope("model",reuse=True):
			m_test=Model(kind_size,vocab_size,seq_length,batchsize=batchsize,keep_prob=1.0,is_training=False)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		if recover:
			print("restore model")
			model_file=tf.train.latest_checkpoint('my-model/')
			saver.restore(sess,model_file)
		for epoch in range(current_epoch+1,epochs):
			print("epoch:{}".format(epoch))
			flag=0
			for kind_data,desc_data,y_data in get_data(batchsize,kind_size,seq_length):
				# pdb.set_trace()
				# batch_tensor=np.zeros(y_data.shape,dtype="int32")
				m.train_step.run(feed_dict={m.input_kind:kind_data,m.input_content:desc_data,m.y_label:y_data})
				value=m.loss.eval(feed_dict={m.input_kind:kind_data,m.input_content:desc_data,m.y_label:y_data})
				flag+=1
				if flag%1000==0:
					print('loss: {}'.format(value))
			saver.save(sess, 'my-model/my-model', global_step=epoch)
			print("test epoch {}".format(epoch))
			#write_log("test epoch {}".format(epoch),epoch)
			for kind_test,desc_test in get_data(batchsize,kind_size,seq_length,'test',False):
				result=m_test.result.eval(feed_dict={m_test.input_kind:kind_test,m_test.input_content:desc_test})
				for v in result:
					val=''
					for v1 in  v:
						val+=str(v1)+' '
					write_log(val,epoch)

if __name__ == "__main__":
	seq_length=100
	kind_size=387
	vocab_size=10217
	prep=Preprocess()
	if not os.path.exists("data/traindata.json"):
		prep.word_id("train_set.csv","train")
	if not os.path.exists("data/testdata.json"): 
		prep.word_id("test_set.csv","test")
	if not os.path.exists("data/train.json"):
		store_data(maxlen=seq_length)
	if not os.path.exists("data/test.json"):
		store_data("test",maxlen=seq_length)
	train_model(seq_length,kind_size,vocab_size,recover=True,current_epoch=0)
