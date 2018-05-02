# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import math
import os
import pdb
import json
import jieba
import pandas as pd
import numpy as np


def cal_data(flag="train",maxlen=0):
	if not os.path.exists("data/{}data.json".format(flag)):
		print("{}data has not existed!".format(flag))
		return
	res=0
	with open("data/{}data.json".format(flag)) as f:
		for _ in f.readlines():
			_=json.loads(_)
			vals=list(_.values())[0]
			kind=[]
			descrip=[]
			for i in range(387):
				if len(vals[str(i)])<=maxlen:
					continue
				else:
					res+=1
	print(res)

def store_data(flag="train",maxlen=100):
	if not os.path.exists("data/{}data.json".format(flag)):
		print("{}data has not existed!".format(flag))
		return
	res=[]
	with open("data/{}data.json".format(flag)) as f:
		for _ in f.readlines():
			_=json.loads(_)
			vals=list(_.values())[0]
			kind=[]
			descrip=[]
			for i in range(387):
				kind.append(i)
				temp=[]
				if len(vals[str(i)])<maxlen:
					temp=vals[str(i)]+[0]*(maxlen-len(vals[str(i)]))
				else:
					temp=vals[str(i)][:maxlen]
				descrip.append(temp)
			with open("data/{}.json".format(flag),'a') as f:
				f.write(json.dumps([kind,descrip,vals['res']])+'\n')


def get_data(batchsize,kind_size,seq_length,flag="train",shuffle=True):
	if not os.path.exists("data/{}data.json".format(flag)):
		print("traindata has not existed!")
		return
	with open("data/{}.json".format(flag)) as f:
		lines=f.readlines()
		indices=np.arange(len(lines))
		if shuffle:
			np.random.shuffle(indices)
		#pdb.set_trace()
		for start_index in range(0,len(lines)-batchsize+1,batchsize):
			sub_list=indices[start_index:start_index+batchsize]
			kind=np.zeros((batchsize,kind_size)).astype('int32')
			desc=np.zeros((batchsize,kind_size,seq_length)).astype('int32')
			if flag=="train":
				y=np.zeros((batchsize,5)).astype('float32')
			for i,index in enumerate(sub_list):
				_=json.loads(lines[index])
				kind[i,]=_[0]
				desc[i,]=_[1]
				if flag=="train":
					y[i,]=_[2]
			if flag=="train":
				yield kind,desc,y
			else:
				yield kind,desc
		start_index+=batchsize
		sub_list=indices[start_index:len(lines)]
		kind=np.zeros((len(lines)-start_index,kind_size)).astype('int32')
		desc=np.zeros((len(lines)-start_index,kind_size,seq_length)).astype('int32')
		if flag=="train":
			y=np.zeros((len(lines)-start_index,5)).astype('float32')
		for i,index in enumerate(sub_list):
			_=json.loads(lines[index])
			kind[i,]=_[0]
			desc[i,]=_[1]
			if flag=="train":
				y[i,]=_[2]
		if flag=="train":
			yield kind,desc,y
		else:
			yield kind,desc
	

def get_test_data(flag="test"):
        if not os.path.exists("data/{}data.json".format(flag)):
                print("testdata has not existed!")
                return
        input_kind=[]
        input_descrip=[]
        with open("data/{}.json".format(flag)) as f:
                for _ in f.readlines():
                        _=json.loads(_)
                        #pdb.set_trace()
                        input_kind.append(_[0])
                        input_descrip.append(_[1])
        return input_kind,input_descrip


class Preprocess(object):

	def __init__(self,folder='data',train_file="train_set.csv",test_file="test_set.csv"):
		self.folder=folder
		self.train_file=train_file
		self.test_file=test_file

	def para_val(self,word):
		"""分段处理数字"""
		val=float(word)
		val=math.floor(val)
		res=val
		if val<100:
			res=val
		elif val<1000:
			res=10*int(val/10)
		elif val<10000:
			res=100*int(val/100)
		else:
			res="nan"
		return str(res)

	def deal_x(self,s,vocab):
		"""是的，就是为了处理x和%"""
		flag=0
		i=0
		length=len(s)
		res=[]
		while i<length:
			if s[i]=='x' or s[i]=="%":
				if flag<i:
					res.append(str(s[flag:i]))
				res.append(s[i])
				i+=1
				flag=i
			else:
				i+=1
		if flag<i:
			res.append(str(s[flag:i]))
			# print(res)
		for word in res:
			try:
				word=self.para_val(word)
			except:
				pass
			if word=="nan":
				continue
			try:
				vocab[word]+=1
			except:
				vocab[word]=1


	def add_x(self,s,vocab,temp):
		"""是的，就是为了把x和%加到数据里"""
		flag=0
		i=0
		length=len(s)
		res=[]
		while i<length:
			if s[i]=='x' or s[i]=="%":
				if flag<i:
					res.append(str(s[flag:i]))
				res.append(s[i])
				i+=1
				flag=i
			else:
				i+=1
		if flag<i:
			res.append(str(s[flag:i]))
			# print(res)
		for word in res:
			try:
				word=self.para_val(word)
			except:
				pass
			if word=="nan":
				continue
			try:
				temp.append(vocab[word])
			except:
				temp.append(0)

	def kind_id(self):
		if os.path.exists("{}/kinds_id.txt".format(self.folder)):
			print("kinds_id completed!")
			# return
		f=pd.read_csv("{}/{}".format(self.folder,self.train_file),encoding='utf-8',low_memory=False)
		# pdb.set_trace()
		length=len(f.columns)
		res=[]
		pdb.set_trace()
		for i in range(6,length):
			res.append("{} {}".format(f.columns[i],i-6))
		with open("{}/kinds_id.txt".format(self.folder),'a') as f:
			for _ in res:
				f.write(_+'\n')


	def count_vocab(self):
		vocab={}
		res=[]
		if os.path.exists("{}/vocab_id.txt".format(self.folder)):
			print("vocab_id has existed!")
			with open("{}/vocab_id.txt".format(self.folder)) as f:
				for _ in f.readlines():
					_=_.split()
					vocab[_[0]]=_[1]
			for i in range(101):
				try:
					a=vocab[str(i)]
				except:
					res.append("{} {}\n".format(i,1))
			for i in range(110,1001,10):
				try:
					a=vocab[str(i)]
				except:
					res.append("{} {}\n".format(i,1))
			for i in range(1100,10000,100):
				try:
					a=vocab[str(i)]
				except:
					res.append("{} {}\n".format(i,1))
			with open("{}/vocab_id.txt".format(self.folder),'a') as f:
				for _ in res:
					f.write(_)
			return
		
		f_train=pd.read_csv("{}/{}".format(self.folder,self.train_file),encoding='utf-8',low_memory=False)
		f_test=pd.read_csv("{}/{}".format(self.folder,self.test_file),encoding='utf-8',low_memory=False)
		f=pd.concat([f_train,f_test],axis=0)
		jieba.load_userdict("data/dict.txt")
		for v in f.values:
			for v1 in v[6:]:
				if v1:
					v1=jieba.lcut(str(v1).lower())
					for word in v1:
						word=word.strip()
						if not word:
							continue
						if ('x'!=word and '%'!=word) and ('x' in word or '%' in word):
							self.deal_x(word, vocab)
							continue
						try:
							word=self.para_val(word)
						except:
							pass
						if word=="nan":
							continue
						try:
							vocab[word]+=1
						except:
							vocab[word]=1
		vocab=sorted(vocab.items(), key=lambda d: d[1],reverse=True)
		print(len(vocab))

		with open("{}/vocab_id.txt".format(self.folder),'a') as f:
			i=1
			for k,v in vocab:
				f.write("{} {} {}\n".format(k,v,i))
				i+=1

	def word_id(self,file,aim):
		"""把数据集数字化"""
		kind2id={}
		vocab={}
		if not os.path.exists("{}/kinds_id.txt".format(self.folder)):
			print("kinds2id has not existed!")
			return
		if not os.path.exists("{}/vocab_id.txt".format(self.folder)):
			print("vocab2id has not existed!")
			return
		with open("{}/kinds_id.txt".format(self.folder)) as f:
			for _ in  f.readlines():
				_=_.split()
				kind2id[_[0]]=int(_[1])
		with open("{}/vocab_id.txt".format(self.folder)) as f:
			for _ in  f.readlines():
				_=_.split()
				try:
					vocab[_[0]]=int(_[2])
				except:
					pdb.set_trace()
		res=[]
		f=pd.read_csv("{}/{}".format(self.folder,file),encoding='utf-8',low_memory=False)
		jieba.load_userdict("data/dict.txt")
		maxlen=0
		for v in f.values:
			temp={}
			temp[v[0]]={}
			temp[v[0]]['res']=list(v[1:6])
			for i in range(6,len(v)):
				v1=v[i]
				temp[v[0]][str(i-6)]=[]
				if v1:
					v1=jieba.lcut(str(v1).lower())
					for word in v1:
						word=word.strip()
						if not word:
							continue
						if ('x'!=word and '%'!=word) and ('x' in word or '%' in word):
							self.add_x(word,vocab,temp[v[0]][str(i-6)])
							continue
						try:
							word=self.para_val(word)
						except:
							pass
						if word=="nan":
							continue
						try:
							temp[v[0]][str(i-6)].append(vocab[word])
						except:
							# pdb.set_trace()
							temp[v[0]][str(i-6)].append(vocab["UNK"])
					maxlen=max(maxlen,len(temp[v[0]][str(i-6)]))
			res.append(temp)
		print(maxlen)
		# pdb.set_trace()
		with open("{}/{}data.json".format(self.folder,aim),'a') as f:
			for _ in res:
				f.write(json.dumps(_)+'\n')

	def zero_pad(self):
		if not os.path.exists("{}/traindata.json".format(self.folder)):
			print("traindata has not existed!")
			return
		if not os.path.exists("{}/testdata.json".format(self.folder)):
			print("testdata has not existed!")
			return
		with open("{}/traindata.json".format(self.folder)) as f:
			for _ in f.readlines():
				f.write(json.dumps(_)+'\n')


if __name__=="__main__":
	#a,b=get_data(50,387,500)
	#prep=Preprocess()
	#prep.kind_id()
	# prep.count_vocab()
	# prep.word_id("train_set.csv","train")
	# prep.word_id("test_set.csv","test")
	# vocab={}
	# prep.deal_x("19x13mm",vocab)
	get_data(file='test')
	#cal_data()

