import numpy as np
#from sklearn.utils.graph import graph_shortest_path
from scipy import sparse
def loadDataSet(filename):
    data = np.loadtxt(filename, delimiter = ',')
    dataSet = [list[0:-1] for list in data]
    labelSet = [list[-1] for list in data]
    return np.array(dataSet), labelSet

def knn_Graph(dataSet, topNfeat=99999):
    distance = []
    sortdistance = []
    dataSet = np.array(dataSet)
    dataSetSize = dataSet.shape[0]
    graph = np.ones((dataSetSize, dataSetSize)) * float('inf')
    for i in range(dataSetSize):
        distance = [np.linalg.norm(dataSet[i]-dataSet[j]) for j in range(dataSetSize)]
        sortdistance = np.array(distance).argsort()
        for j in range(topNfeat+1):
            k = sortdistance[j]
            graph[i][k] = distance[k]
        distance = []
    return graph

def isomap(graphMat):
    shortestMat = sparse.csgraph.shortest_path(graphMat,directed=False)
    return shortestMat

def mds(dataMat, target, disMat):
    dataMatsize = len(dataMat)
    if target > dataMatsize:
        target = dataMatsize
    distij = 0.0
    disti = np.zeros([dataMatsize],np.float32)
    distj = np.zeros([dataMatsize],np.float32)
    matA = np.zeros([dataMatsize,dataMatsize], np.float32)
    for idx in range(dataMatsize):
        for subidx in range(dataMatsize):
            dist = np.square(disMat[idx][subidx])
            distij += dist
            disti[idx] += dist
            distj[subidx] += dist / dataMatsize
        disti[idx] /= dataMatsize
    distij /= np.square(dataMatsize)
    for idx in range(dataMatsize):
        for subidx in range(dataMatsize):
            dist = np.square(disMat[idx][subidx])
            matA[idx][subidx] = -0.5 * (dist - disti[idx] - distj[subidx] + distij)
    a,v = np.linalg.eig(matA)
    listidx = np.argpartition(a, target-1)[-target:]
    a = np.diag(np.maximum(a[listidx],0.0))
    x = np.matmul(v[:, listidx], np.sqrt(a))
    return np.matmul(v[:, listidx], np.sqrt(a))


def oneNN(inx, dataSet, labels):
    distance = []
    sortdistance = []
    dataSetSize = dataSet.shape[0]
    inxSize = inx.shape[0]
    inx = np.array(inx)
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    for i in range(inxSize):
        tdistance = [np.linalg.norm(inx[i]-dataSet[j]) for j in range(dataSetSize)]
        distance.append(tdistance)
    for llist in distance:
        llist = np.array(llist)
        sortdistance.append(llist.argsort())
    prelabel = [labels[sortdistance[index][0]] for index in range(inxSize)]
    return np.array(prelabel)

def Test(predictMat, labelsMat):
    dataSize = predictMat.shape[0]
    j=0
    for i in range(dataSize):
        if predictMat[i] == labelsMat[i]:
            j=j+1
    ant = j/dataSize
    return ant


def main():
    a1,a2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-train.txt')
    #a1, a2 = loadDataSet('C:/Users/Asmoc/desktop/splice-train.txt')
    test1,test2 = loadDataSet('C:/Users/Asmoc/desktop/sonar-test.txt')
    #test1, test2 = loadDataSet('C:/Users/Asmoc/desktop/splice-test.txt')
    train_aT = np.vstack([a1, test1])
    split_aT = np.shape(a1)[0]
    tag = 10
    while tag < 31:
        graph = knn_Graph(train_aT, 200)
        l = isomap(graph)
        lowdata = mds(train_aT, tag, l)
        train,test = np.split(lowdata, [split_aT])
        pre = oneNN(test, train, a2)
        num = Test(pre, test2)
        print(tag , ": " , num)
        tag = tag + 10

if __name__ == "__main__":
    main()