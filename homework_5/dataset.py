class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class Dataset(object):
    def __init__(self):
        self.M = 0
        self.V = 0
        self.docs = []
        self.word2id = {}    # <string,int>字典
        self.id2word = {}    # <int, string>字典

    def writewordmap(self, wordmapfile):
        with open(wordmapfile, 'w', encoding = "UTF-8") as f:
            for k,v in self.word2id.items():
                if len(k) > 2:
                    f.write(k + '\t' + str(v) + '\n')



def readtrnfile(trnfile):
    wordmapfile = "dataset/wordbag.txt"
    #print ('Reading train data...')
    with open(trnfile, 'r', encoding = "UTF-8") as f:
        docs = f.readlines()


    dset = Dataset()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split()
            tmp = [tok.lower() for tok in tmp if len(tok)>2]
            #生成一个文档对象
            doc = Document()
            for item in tmp:
                if item in dset.word2id:
                    doc.words.append(dset.word2id[item])
                else:
                    dset.word2id[item] = items_idx
                    dset.id2word[items_idx] = item
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dset.docs.append(doc)
        else:
            pass
    dset.M = len(dset.docs)
    dset.V = len(dset.word2id)
    #print('There are %d documents'%(dset.M))
    #print ('There are %d items' % dset.V)
    #print ('Saving wordmap file...')
    dset.writewordmap(wordmapfile)
    return dset