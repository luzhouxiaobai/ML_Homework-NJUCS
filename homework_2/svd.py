from numpy import *

def loadDataSet(filename):
    data = loadtxt(filename, delimiter=',')
    floatArr = [llist[0:-1] for llist in data]
    classMat = [llist[-1] for llist in data]
    return mat(floatArr), classMat

def svd(dataMat, topNfeat=9999):
    Q,sigma,P = linalg.svd(dataMat)
    lowDataMat = dataMat * transpose(P[:topNfeat,:])
    return lowDataMat, transpose(P[:topNfeat,:]), topNfeat

def svd_test(dataMat, transMat, topNfeat):
    lowTestMat = dataMat * transMat
    return lowTestMat

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
    return ant

def main():
    #a1,a2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-train.txt')
    a1, a2 = loadDataSet('C:/Users/Asmoc/desktop/splice-train.txt')
    tag = 10
    while tag < 31:
        b1,b2,k = svd(a1, tag)
        #test1,test2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-test.txt')
        test1, test2 = loadDataSet('C:/Users/Asmoc/desktop/splice-test.txt')
        l = svd_test(test1, b2, k)
        pre = oneNN(l, b1, a2)
        num = Test(pre, test2)
        print(tag , ": " , num)
        tag = tag + 10

if __name__ == "__main__":
    main()