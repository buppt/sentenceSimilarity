# coding:utf-8
import xlrd
import codecs
import math
import jieba
import pandas as pd
import numpy as np

from compiler.ast import flatten
import pickle
from scipy.linalg import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def xlsx2txt():
    workbook = xlrd.open_workbook('./630T.xlsx')
    for i in workbook.sheet_names():                  #查看所有sheet:
        print i
    booksheet = workbook.sheet_by_index(0)      
      
    #读单元格数据
    cell_11 = booksheet.cell_value(0,0)
    cell_21 = booksheet.cell_value(1,0)
    print cell_11, cell_21, 
    #表总行数
    with codecs.open('./guzhang.txt','w','utf-8') as output:
        for i in range(booksheet.nrows):
            row = booksheet.row_values(i)
            #output.write(row[10]+'\t'+row[11]+'\t'+row[12]+'\n')
            seg_list = jieba.cut(row[10])
            output.write(" ".join(seg_list)+'\n')

def txt2id():
    datas = list()
    
    input_data = codecs.open('./guzhang.txt','r','utf-8')
    for line in input_data.readlines():
        line = line.split()
        datas.append(line)
    input_data.close()
    
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words)+1)
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    word2id["unknow"] = len(word2id)+1
    w2id = {}
    for i in id2word:
        w2id[i] = word2id[i]
    print id2word
    
    '''
    max_len = 20    
    def X_padding(words):
        """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
        ids = list(word2id[words])
        if len(ids) >= max_len:  # 长则弃掉
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        return ids
    
    df_data = pd.DataFrame({'words': datas}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    x = np.asarray(list(df_data['x'].values))

    #print x[0]
    
    with open('./data.pkl', 'wb') as outp:
        pickle.dump(w2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(x, outp)
    '''

def pretrained_embedding():
    with open('./data.pkl', 'rb') as inp:
	    word2id = pickle.load(inp)
    word2vec = {}
    with codecs.open('sgns.wiki.word','r','utf-8') as input_data:   
        for line in input_data.readlines():
            line = line.split()
            if line[0] in word2id: 
                print line[0]
                word2vec[line[0]] = line[1:]
    print len(word2vec)


def vec2(s1,s2):
    '''
    with open('./vec.pkl', 'rb') as inp:
	    word2vec = pickle.load(inp)
    '''
    word2vec = {}
    with codecs.open('./guzhang_embedding.txt','r','utf-8') as input_data:   
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])
    '''   
    with open('./data.pkl', 'rb') as inp:
	    word2id = pickle.load(inp)
	
    for i in word2id:
	    if i in word2vec:
	        continue
	    else:
	        print i  
	
    def sentence_vector(s):
      #  words = jieba.lcut(s)
        words = jieba.cut_for_search(s)
        v = np.zeros(100)
        i = 0
        for word in words:
            if word in word2vec:
                v+=map(float,word2vec[word])
                i+=1
                
       # v /= len(words)
        if i!=0:
            v /= i
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)  
    '''
    
    def softmax(v):
        sumV = 0
        for i in v:
            sumV += i
        
        if sumV==0:
            return 0
        else:
            return v/abs(sumV)
    
    
    '''
    #lstm初始化的参数还是要预训练，不搞了
    hidden_dim = 200
    def hidden_lstm():
        return (torch.randn(2, 1, hidden_dim // 2),
                torch.randn(2, 1, hidden_dim // 2))
    hidden = hidden_lstm()        
    lstm = nn.LSTM(100,hidden_dim / 2,num_layers=1, bidirectional=True)
        
    def lstm_vector(s):
      #  words = jieba.lcut(s)
        words = jieba.cut_for_search(s)
        v = np.zeros(100)

        ids = []
        for word in words:
            if word in word2vec:
                ids.append(map(float,word2vec[word]))

        maxLength = 10
        if len(ids)>maxLength:
            ids = ids[:maxLength]
        else:
            while len(ids)<maxLength:
                ids.append(v) 
             
        ids = np.asarray(ids)
        
        ids =torch.from_numpy(ids).float().view(10,1,-1)
        #print ids.size()
        ids, (h_n, c_n) = lstm(ids,hidden)
       # print ids.size()
        ids = ids.view(10,-1).data.numpy()
        res = np.zeros(hidden_dim)
        for i in ids:
            res+=i
        return softmax(res)

    v1, v2 = lstm_vector(s1), lstm_vector(s2)  
    '''
    if norm(v1) * norm(v2)==0:
        return 0    
    else:  
        return np.dot(v1, v2) / (norm(v1) * norm(v2))




s1 =[ '安全门自动落下不停止',
'安全门打不开',
'右侧安全门无法打开',
'电机异常断电',
'设备无动作',
'送料机不送料',
'工作台不能移入',
'中间废料门打开限位异常',
'送料机经常送料无动作',
'送料机升不起来',
'送料机丝杆不能升降',
'送料机高度不够',
'后提升门上升无动作',
'通讯模块多次空开跳闸',
'后安全门下限位无效']
s2 = '送料机丝杆不能升降'
for i in s1:
    print i, vec2(i, s2)

