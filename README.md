NTM���а���ntm��

ntm�������º�����
entityRecognition(self,string)δʵ��
���ڴ�ʵ����л�ȡstring�Ķ�Ӧʵ��

getBaseData(self)δʵ��
���ڴ����ݿ��л�ȡ����ѵ��ʹ�õ����ݼ�

saveBaseData(self��name)
��Ŀǰʹ�õ����ݼ����浽����(csv��ʽ)

loadBaseData(self,name)
��ȡ���ص����ݼ���csv��

loadStopWord(self,path)
��ȡstopword(txt��ÿ��һ��)

loadWord2Vec(self,VECTOR_DIR)
��ȡword2vec�������ļ�
���ÿ���������ĳ���

dataPreProgress(self,ori,rate)
ori�����ݼ�
rate��ÿ�����ﱻʹ�õĴ���
�ڱ������У�ÿƪ���ŵ�ÿ����stopword�ᱻת��Ϊ��Ӧ���������浽data\ngramX
�����Ӧ��������ű��浽data\posX
�����ȡrateƪ�������ô�������£����浽data\negX

loadTensor(self,n,length)
��ngramX��posX��negX�е�����ת��Ϊ�����������
nΪ��Ӧ��X
lengthΪ������������ͨ��BaseData���

generateNN(self,numOfStoryLine)
����������
numOfStoryLine��storyline��������

meanLoss()
��������Զ�����ʧ����������batch����ʧ��ƽ��ֵ

sumLoss()
��������Զ�����ʧ����������batch����ʧ�ĺ�

composeNN(self,lr=0.01,loss=None)
����������

fitNN(self)
ѵ�������粢����

loadNN(self,name = None)
���ر��ر����������

predict(self)
���ÿƪ���Ŷ�Ӧstoryline��Ȩ��

weight2storyline(self)
��Ȩ��ת��Ϊstoryline��Ų�д��BaseData��

run()
������