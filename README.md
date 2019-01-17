NTM包中包含ntm类

ntm类有以下函数：
entityRecognition(self,string)未实现
用于从实体库中获取string的对应实体

getBaseData(self)未实现
用于从数据库中获取本轮训练使用的数据集

saveBaseData(self，name)
将目前使用的数据集保存到本地(csv格式)

loadBaseData(self,name)
读取本地的数据集（csv）

loadStopWord(self,path)
读取stopword(txt，每行一个)

loadWord2Vec(self,VECTOR_DIR)
读取word2vec词向量文件
获得每个词向量的长度

dataPreProgress(self,ori,rate)
ori：数据集
rate：每个词语被使用的次数
在本函数中，每篇新闻的每个非stopword会被转化为对应词向量保存到data\ngramX
词语对应的文章序号保存到data\posX
随机抽取rate篇不包含该词语的文章，保存到data\negX

loadTensor(self,n,length)
将ngramX、posX、negX中的数据转化为神经网络的输入
n为对应的X
length为文章总数，可通过BaseData获得

generateNN(self,numOfStoryLine)
生成神经网络
numOfStoryLine：storyline的最大个数

meanLoss()
神经网络的自定义损失函数，采用batch的损失的平均值

sumLoss()
神经网络的自定义损失函数，采用batch的损失的和

composeNN(self,lr=0.01,loss=None)
编译神经网络

fitNN(self)
训练神经网络并保存

loadNN(self,name = None)
加载本地保存的神经网络

predict(self)
获得每篇新闻对应storyline的权重

weight2storyline(self)
将权重转化为storyline编号并写入BaseData中

run()
测试用