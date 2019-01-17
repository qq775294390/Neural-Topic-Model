# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gensim
import random
from tensorflow import keras
K = keras.backend

class ntm:
    def __init__(self):
        self.basedata = None
        self.sizeOfData = None
        self.NN = None
        self.stopword = None
        self.w2v = None
        self.dimensionOfW2V = None
        self.pos_dic = None
        self.neg_dic = None
        self.pos = None
        self.neg = None
        self.ngram = None
        self.model = None
    def entityRecognition(self, string):
        # get entity of string
        # to be completed

        return string

    def getBaseData(self):
        # get basedata form databases
        # to be completed
        self.basedata = None
        return

    def saveBaseData(self,name='data.csv'):
        if self.basedata!=None:
            self.basedata.to_csv(name)
        else:
            print('No basedata')
        return

    def loadBaseData(self, name='data.csv'):
        self.basedata = pd.read_csv(name)
        self.sizeOfData = len(self.basedata)

    def loadStopWord(self,path='stopword.txt'):
        with open(path, encoding='UTF-8') as f:
            self.stopword = f.read().split()
        return

    def loadWord2Vec(self,VECTOR_DIR='vectors.bin'):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
        self.dimensionOfW2V = len(self.w2v['a'])
        return

    def dataPreProgress(self,ori=None,rate=5):
        if ori==None:
            if self.basedata==None:
                self.loadBaseData()
            ori=self.basedata
        if self.stopword==None:
            self.loadStopWord()
        if self.w2v==None:
            self.loadWord2Vec()
        content = ori['content']
        pos_dic = {}
        neg_dic = {}

        a = np.zeros((len(content)))
        print('posPreProgress')
        for i, s in enumerate(content):
            #     a[i]=len(j.split())
            if i%100==0:
                print(i)
            for word in s.split():
                if word in self.stopword:
                    continue
                if word not in pos_dic:
                    pos_dic[word] = {}
                if i not in pos_dic[word]:
                    pos_dic[word][i] = 0
                pos_dic[word][i] += 1
        newsStopword = {}
        low = {}
        print('del new stopwords')
        for word in list(pos_dic.keys()):
            if len(pos_dic[word]) > 500:
                newsStopword[word] = len(pos_dic[word])
                del pos_dic[word]
            elif len(pos_dic[word]) <= 1:
                low[word] = len(pos_dic[word])
                del pos_dic[word]

        flag=0
        print('negPreProgress')
        for word in pos_dic:
            flag+=1
            if flag%100==0:
                print (flag)
            neg_dic[word] = {}
            while len(neg_dic[word]) < rate * len(pos_dic[word]):
                import random
                rand = random.randrange(len(ori))
                if rand not in pos_dic[word]:
                    if rand not in neg_dic[word]:
                        neg_dic[word][rand] = 0
                    neg_dic[word][rand] += 1
        ngram = []
        pos = []
        neg = []
        flag = 0
        for word in pos_dic:
            flag += 1
            if flag % 100 == 0:
                print(flag)

            if word not in self.w2v:
                continue
            pos += rate * list(pos_dic[word].keys())
            neg += list(neg_dic[word].keys())
            ngram += [self.w2v[word]] * len(neg_dic[word])

        del pos_dic
        del neg_dic
        count = max(max(neg), max(pos)) - min(min(neg), min(pos)) + 1

        temp = list (zip(pos, neg, ngram))
        import random
        random.shuffle(temp)
        pos, neg, ngram = list(zip(*temp))

        flag = 0
        print('saveIO')
        for i in range(len(pos) // 10000):

            flag += 1
            if flag % 10 == 0:
                print(flag)

            with open('data/neg' + str(i), 'w') as f:
                f.write(str(neg[i * 10000:(i + 1) * 10000]))
            with open('data/pos' + str(i), 'w') as f:
                f.write(str(pos[i * 10000:(i + 1) * 10000]))
            np.savetxt('data/ngram' + str(i), ngram[i * 10000:(i + 1) * 10000])
        return
    def loadTensor(self,n,length=None):
        if length==None:
            length = self.sizeOfData
        with open('data/pos' + str(n)) as f:
            pos = eval(f.read())
        with open('data/neg' + str(n)) as f:
            neg = eval(f.read())
        ngram = np.loadtxt('data/ngram' + str(n))
        input_pos = np.zeros((len(pos), length))
        for i in range(len(pos)):
            input_pos[i][pos[i]] = 1

        input_neg = np.zeros((len(pos), length))
        for i in range(len(neg)):
            input_neg[i][neg[i]] = 1
        input_ngram = np.array(ngram)
        return input_pos, input_neg, input_ngram
    def generateNN (self,numOfStoryline=50):
        if self.sizeOfData==None:
            self.loadBaseData()
        if self.w2v==None:
            self.loadWord2Vec()
        input_ngram = keras.layers.Input((self.dimensionOfW2V,))
        out_ngram = keras.layers.Dense(numOfStoryline, activation='sigmoid', use_bias=False,
                                       kernel_regularizer=keras.regularizers.l2(0.001))(input_ngram)
        input_pos = keras.layers.Input((self.sizeOfData,))
        input_neg = keras.layers.Input((self.sizeOfData,))
        den_d = keras.layers.Dense(numOfStoryline, activation='softmax', use_bias=False,
                                   kernel_regularizer=keras.regularizers.l1(0.001))
        out_pos = den_d(input_pos)
        out_neg = den_d(input_neg)
        dot = keras.layers.Dot(axes=-1, normalize=True)
        dot_pos = dot([out_pos, out_ngram])
        dot_neg = dot([out_neg, out_ngram])
        out = keras.layers.Concatenate(axis=-1)([dot_pos, dot_neg])
        self.model = keras.models.Model(inputs=[input_pos, input_neg, input_ngram], outputs=out)
        self.model_out = keras.models.Model(inputs=input_pos, outputs=out_pos)
        return

    def meanLoss(self,y_true, y_pred):
        temp = y_pred[:, 1] - y_pred[:, 0] + 0.5

        return K.mean(K.relu(temp))

    def sumLoss(self,y_true, y_pred):
        temp = K.relu(y_pred[:, 1] - y_pred[:, 0] + 0.5)
        return K.sum(temp)

    def composeNN(self,lr=0.01,loss=None):
        if self.model==None:
            self.generateNN()
        if loss==None:
            loss=self.sumLoss
        sgd = keras.optimizers.SGD(0.01)
        self.model.compile(loss=loss, optimizer=sgd)
        return
    def fitNN(self):
        self.composeNN()
        y_true = np.array([[-1, 1]] * 10000)
        for k in range(2):
            for i in range(300):
                print(i)
                pos, neg, ngram = self.loadTensor(i)
                self.model.fit([pos, neg, ngram], y_true, epochs=10, batch_size=1000, )
        self.model_out.save('ntm_out.h5')
        self.model.save('ntm.h5')
        return
    def loadNN(self,name = None):
        if name==None:
            name='ntm_out.h5'
        self.model_out = keras.models.load_model(name)
        return

    def predict(self):
        onehot = np.zeros((self.sizeOfData, self.sizeOfData))
        for i in range(self.sizeOfData):
            onehot[i][i] = 1
        self.weight = self.model_out.predict(onehot)
    def weight2storyline(self):
        res = []
        support = []
        if self.weight is None:
            self.predict()
        for i in range(self.sizeOfData):

            temp = self.weight[i].argmax()
            res.append(temp)
            support.append( self.weight[i][temp] )
        self.basedata['storyline']=res
        self.basedata['support']=support
    def run(self):
        # self.dataPreProgress()
        self.fitNN()
        self.predict()
        self.weight2storyline()

