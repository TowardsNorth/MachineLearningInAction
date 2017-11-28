'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():  #创建训练集
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #分类的标签，1表示侮辱性言论，0表示不是
    return postingList,classVec   #返回训练集和标签
                 
def createVocabList(dataSet):   #创建一个词汇表
    vocabSet = set([])  #create empty set，set集合，元素会唯一
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets，“\”表示并操作
    return list(vocabSet)  #返回vocabSet列表集合

def setOfWords2Vec(vocabList, inputSet): #将输入的集合转化为向量，词集模型
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  #这里的等于1表示词集模型，表示将该词出现与否作为一个特征。还有一种叫词袋模型，表示会记录该单词出现的次数
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec    #返回向量

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)   #训练集有多少条
    numWords = len(trainMatrix[0])    #trainMatrix[0]的长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #训练集中是侮辱性的言论所占的比例
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() ，因为这是一个文本分析，有些单词没有出现就会变成0，为了避免这种影响，全都初始化为1
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0，分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   #这个值为1，表示这是一个侮辱性的言论
            p1Num += trainMatrix[i] #p1Num这是一个向量，这里表示向量相加
            p1Denom += sum(trainMatrix[i])   #将总和相加起来
        else:                       #这里表示不是侮辱性的言论
            p0Num += trainMatrix[i]   #向量相加
            p0Denom += sum(trainMatrix[i])   #将总和相加起来
    p1Vect = log(p1Num/p1Denom)          #change to log() ，加log的目的是防止下溢，因为几个很小的数相乘其结果可能就会变成0
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive  #返回训练出来的分类器的向量

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): #贝叶斯分类器
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:  #如果p1>p0.则是类别1
        return 1
    else:    #否则是类别0
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):   #词袋模型
    returnVec = [0]*len(vocabList)  #创建一个向量，初始化为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec  #返回一个向量

def testingNB():   #测试函数
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) #返回训练出来的概率分布
    testEntry = ['love', 'my', 'dalmation']  #测试数据
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  #将词转化为向量,array函数是将列表转化为数组
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))  #输出测试结果
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))   #输出测试结果

def textParse(bigString):    #input is big string, #output is word list  文本解析
    import re                #正则表达式
    listOfTokens = re.split(r'\W*', bigString)  #切割文本
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  #返回长度大于2，且是小写的字符串
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())  #获取指定目录下的文件来进行解析
        docList.append(wordList)  #加入到docList中
        fullText.extend(wordList)  #加入到fullText中
        classList.append(1)      #设置该文本的标签，表示这是侮辱性的言论，置1
        wordList = textParse(open('email/ham/%d.txt' % i).read())   #获取指定目录下的文件来进行解析
        docList.append(wordList)  #加入到docList中
        fullText.extend(wordList)  #加入到fullText中
        classList.append(0)      #设置该文本的标签，表示这不是侮辱性的言论，置0
    vocabList = createVocabList(docList)#create vocabulary  创建词汇表
    trainingSet = range(50); testSet=[]           #create test set，创建测试集
    for i in range(10):      #随机生成10个测试集
        randIndex = int(random.uniform(0,len(trainingSet)))  #随机生成一个下标索引
        testSet.append(trainingSet[randIndex])   #将该索引加入到测试集中
        del(trainingSet[randIndex])    #删除训练集中该索引对应的项，删除的只是一个下标
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0，用来训练，全部样本中，去除测试集，剩下的是训练集，训练集与测试集不交叉，这叫做交叉保留验证
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  #将得到的向量加入到训练集中
        trainClasses.append(classList[docIndex])          #将对应的训练集的标签加入到trainClasses中
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))  #训练之后得到分类的概率值
    errorCount = 0         #初始化错误值为0
    for docIndex in testSet:        #classify the remaining items，这段代码主要的作用是进行测试
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])      #将字符串转化为向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:  #如果分类错误
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))   #计算错误率
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):  #统计在fulltext中出现频率最高的30个单词
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) #从大到小排序
    return sortedFreq[:30]       #返回在fullText中出现频率最高的30个单词

def localWords(feed1,feed0):  #从rss源获取数据，与spamTest类似
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries'])) #获取长度最小的一个
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):  #统计单词出现最多次数的单词
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])


if __name__ == "__main__":    #main函数
    testingNB()

#这是一个朴素贝叶斯的算法，之所以称之为朴素，是因为各个特征都是独立的，每个特征同等重要，这里就涉及到两个假设了
#这个文档的代码涉及到一些需要处理的内容，比如要避免某些概率为0，可采取的措施为初始化为1.为了防止很小的几个数相乘造成下溢出，那么就要取对数
#朴素贝叶斯分类的正式定义如下：

#应用：
# 解决这个问题的方法一般是建立一个属性模型,对于不相互独立的属性,把他们单独处理。例如中文文本分类识别的时候，我们可以建立一个字典来处理一些词组。如果发现特定的问题中存在特殊的模式属性，那么就单独处理。
# 这样做也符合贝叶斯概率原理，因为我们把一个词组看作一个单独的模式，例如英文文本处理一些长度不等的单词，也都作为单独独立的模式进行处理，这是自然语言与其他分类识别问题的不同点。
# 实际计算先验概率时候，因为这些模式都是作为概率被程序计算，而不是自然语言被人来理解，所以结果是一样的。
# 在属性个数比较多或者属性之间相关性较大时，NBC模型的分类效率比不上决策树模型。但这点有待验证，因为具体的问题不同，算法得出的结果不同，同一个算法对于同一个问题，只要模式发生变化，也存在不同的识别性能。这点在很多国外论文中已经得到公认，在机器学习一书中也提到过算法对于属性的识别情况决定于很多因素，例如训练样本和测试样本的比例影响算法的性能。
# 决策树对于文本分类识别，要看具体情况。在属性相关性较小时，NBC模型的性能稍微良好。属性相关性较小的时候，其他的算法性能也很好，这是由于信息熵理论决定的。




