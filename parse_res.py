# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'

import pdb
import pandas as pd

class Parse(object):
	"""docstring for Parse"""
	def __init__(self, res_file='test_epoch0.log',test_file="data/test_set.csv"):
		super(Parse, self).__init__()
		self.res_file = res_file
		self.test_file=test_file

	def parse(self):
		f_test=pd.read_csv(self.test_file,encoding='utf-8',low_memory=False)
		self.data=pd.DataFrame()
		l=['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']
		vids=f_test['vid'].tolist()
		res=[[],[],[],[],[]]
		#pdb.set_trace()
		with open(self.res_file) as f:
			for _ in f.readlines():
				_=_.split()
				for i in range(5):
					res[i].append(_[i])
		self.data['vid']=vids
		pdb.set_trace()
		for i in range(5):
			self.data[l[i]]=res[i]
		self.data.to_csv('result.csv',index=False,headers=False)

if __name__=="__main__":
	Par=Parse()
	Par.parse()
