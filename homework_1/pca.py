from numpy import *

def loadDataSet(filename):
    data = loadtxt(filename, delimiter=',')
    floatArr = [llist[0:-1] for llist in data]
    classMat = [llist[-1] for llist in data]
    return mat(floatArr), classMat

def pca(dataMat, topNfeat=99999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    #reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, redEigVects, meanVals            #返回降维后的矩阵和投影矩阵

def pca_test(dataMat, transMat, meanVal):               #dataMat是loadDataSet返回的值，transMat是投影矩阵
    #meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVal
    lowTestData = meanRemoved * transMat
    return lowTestData

def oneNN(inx, dataSet, labels):
    distance = []
    sortdistance = []
    dataSetSize = dataSet.shape[0]
    inxSize = inx.shape[0]
    inx = array(inx)
    dataSet = array(dataSet)
    labels = array(labels)
    for i in range(inxSize):
        tdistance = [linalg.norm(inx[i]-dataSet[j]) for j in range(dataSetSize)]
        distance.append(tdistance)
    for llist in distance:
        llist = array(llist)
        sortdistance.append(llist.argsort())
    prelabel = [labels[sortdistance[index][0]] for index in range(inxSize)]
    return array(prelabel)

def Test(predictMat, labelsMat):
    dataSize = predictMat.shape[0]
    j=0
    for i in range(dataSize):
        if predictMat[i] == labelsMat[i]:
            j=j+1
    ant = j/dataSize
    #print(j)
    return ant


def main():
    a1,a2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-train.txt')
    #a1, a2 = loadDataSet('C:/Users/Asmoc/desktop/splice-train.txt')
    tag = 10
    while tag < 31:
        b1,b2,mean = pca(a1, tag)
        test1,test2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-test.txt')
        #test1, test2 = loadDataSet('C:/Users/Asmoc/desktop/splice-test.txt')
        l = pca_test(test1, b2, mean)
        pre = oneNN(l, b1, a2)
        num = Test(pre, test2)
        print(tag , ": " , num)
        tag = tag + 10

if __name__ == "__main__":
    main()