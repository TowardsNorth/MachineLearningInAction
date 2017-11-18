from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):   #计算其距离,knn的核心，基于欧拉距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     #argsort返回的是数组从小到大排序的索引值，note that index
    classCount={}          
    for i in range(k):  #找出k个近邻中，对象所属类别频率最高的那一个
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #从大到小排序
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


#将字符串文本转换为numpy的解析程序
def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return，matrix is equal to numberOfLines*3
    classLabelVector = []                       #prepare labels return   
    index = 0
    for line in arrayOLines:
        line = line.strip()      #截取掉所有的回车字符
        listFromLine = line.split('\t')   #将整行数据分隔成一个元素列表
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector  #classLabelVector是标签

    
def autoNorm(dataSet):    #进行归一化，将取值范围处理为0~1或者-1~1之间，此函数是将数据处理为0~1之间
    minVals = dataSet.min(0)   #获取矩阵中列的最小值，规模为1*3
    maxVals = dataSet.max(0)   #获取矩阵中列的最大值，规模为1*3
    ranges = maxVals - minVals  #最小值与最大值之差，规模为1*3
    normDataSet = zeros(shape(dataSet))   #构造一个与dataSet维数相同的矩阵，并初始化为0
    m = dataSet.shape[0]   #获取矩阵dataSet的行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: %s" % resultList[classifierResult - 1])
    
def img2vector(filename):    #图像文件转换为一维向量
    returnVect = zeros((1,1024))    #初始化一个一维向量
    fr = open(filename)     #打开文件
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect    #返回1*1024的一维向量

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set,列出给定目录的文件名
    m = len(trainingFileList)         #得出当前的文件夹中有多少个文件
    trainingMat = zeros((m,1024))     #构造出m*1024的矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]     #获取到文件名
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])   #获取到文件的标签
        hwLabels.append(classNumStr)      #将标签加入集合
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)   #将图像文件转化为向量，python的语法
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()




k近邻算法：这是一个有训练集有标签的分类算法，其主要的原理在于，计算待分类的物品与训练集中的所有物品的距离（一般为欧拉距离），然后从其中选出距离
最近的K（k）个样本，然后从这个k个选出来的样本中，其所属类别频率最高的即为待分类物品的标签。一般情况下，k近邻算法处理数据时要进行归一化，即将样
本值处在-1~1或者0~1之间。k近邻算法包括训练集和测试集。

knn方法虽然从原理上也依赖于极限定理，但在类别决策时，只与极少量的相邻样本有关。由于knn只依靠周围的有限的样本，而不是靠判别类域的方法来进行分类，因此
对于样本重叠或者交叉过多的场合，knn算法比其它方法更合适。

k近邻算法实际采用的模型实际上对应于特征空间的划分。k值的选择，距离度量，决策规则则是该算法的三大要素。
（1）k值的选择，如果选择的k值较小，则容易产生过拟合。如果K值过大，优点是可以减少学习的估计误差，缺点是距离待测样本较远的训练实例也会起到一个预测的作用，
结果可能使预测错误。在实际中，k值一般选一个较小的值，一般使用交叉验证法来找一个最优的k值
（2）该算法中的决策规格一般采用多数表决
（3）该算法的决策度量一般采用欧式距离，在这一个步骤中，一般需要对原始数据进行归一化，来避免某些属性值过大导致带来较大的权重值，归一化一般化为-1~1或者
0~1

knn算法不仅可以用于分类，也可以用于回归。通过找出一个样本的最近的K个邻居，计算这k个邻居的平均属性，将其赋给样本，即可得到样本的属性值。一个较好的
方法是离待测样本距离远的训练实例赋予一个较小的权重值，距离较近的赋予一个较大的权重值。

knn算法有两个缺点。一是，当样本不平衡时，比如某一类样本很多，另一类样本很少，那么待测实例的k个邻居中，样本多的那一类就占据很大的优势，可以采用权重的
方法来改进。二是，计算的数量很大，因为对于每一个待测样本，都需要计算其与样本实例之间的距离，目前常用的解决方法是事先对已知样本点进行剪辑，事先去除对分类作用不大的样本。
该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较容易产生误分。

knn算法的关键在于如何快速的找出K个邻居。




