# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:28:36 2019

@author: Dcm
"""
"""获取训练数据"""
from config import *

class Poems:
    """poem 类"""
    def __init__(self, filename, isEvaluate=False):
        poems = []
        with open(filename, "r", encoding='utf-8') as f:  
            for line in f:  
                try:
                    # 得到题目 作者 内容
                    title, author, poem = line.strip().split("::")  
                    poem = poem.replace(' ','')
                    # 过滤诗歌内容
                    if len(poem) < 10 or len(poem) > 512:
                        continue
                    if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                        continue
                    poem = '[' + poem + ']' #添加开始和结束标志
                    poems.append(poem) 
                except Exception as e:   
                    pass
        #计算字出现的次数
        wordFreq = collections.Counter()
        for poem in poems:
            wordFreq.update(poem)
        wordFreq[" "] = -1
        wordPairs = sorted(wordFreq.items(), key = lambda x: -x[1])
        self.words, freq = zip(*wordPairs)
        self.wordNum = len(self.words)
        # 每个字映射为一个数字ID  
        self.wordToID = dict(zip(self.words, range(self.wordNum)))
        # 转换为向量形式
        poemsVector = [([self.wordToID[word] for word in poem]) for poem in poems] 
        if isEvaluate: #划分训练集和测试集
            self.trainVector = poemsVector[:int(len(poemsVector) * trainRatio)]
            self.testVector = poemsVector[int(len(poemsVector) * trainRatio):]
        else:
            self.trainVector = poemsVector
            self.testVector = []
        print("训练样本总数： %d" % len(self.trainVector))
        print("测试样本总数： %d" % len(self.testVector))
        
    def generateBatch(self, isTrain=True):
        if isTrain:
            poemsVector = self.trainVector
        else:
            poemsVector = self.testVector

        random.shuffle(poemsVector)
        batchNum = (len(poemsVector) - 1) // batchSize
        X = []
        Y = []
        #create batch
        for i in range(batchNum):
            batch = poemsVector[i * batchSize: (i + 1) * batchSize]
            maxLength = max([len(vector) for vector in batch])
            # 填充到同样长度（最大长度）
            temp = np.full((batchSize, maxLength), self.wordToID[" "], np.int32)
            for j in range(batchSize):
                temp[j, :len(batch[j])] = batch[j]
            X.append(temp)
            temp2 = np.copy(temp)
            temp2[:, :-1] = temp[:, 1:]
            Y.append(temp2)
            """
            X                 Y
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
        return X, Y
    
if __name__ == '__main__':
    poems = Poems(trainPoems)
    X, Y = poems.generateBatch()
