import random, os

class LDAModel(object):
    def __init__(self, dset, K, alpha, beta, iter_num, top_words, word_bags, testdata, finaltopics):
        self.dset = dset

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num
        self.top_words = top_words

        #self.wordmapfile = wordmapfile
        self.wordmapfile = word_bags
        #self.trnfile = trnfile
        self.trnfile = testdata
        #self.modelfile_suffix = modelfile_suffix
        self.modelfile_suffix = finaltopics

        self.p = []        # double类型，存储采样的临时变量
        self.Z = []        # M*doc.size()，文档中词的主题分布
        self.nw = []       # V*K，词i在主题j上的分布
        self.nwsum = []    # K，属于主题i的总词数
        self.nd = []       # M*K，文章i属于主题j的词个数
        self.ndsum = []    # M，文章i的词个数
        self.theta = []    # 文档-主题分布
        self.phi = []      # 主题-词分布

    def init_est(self):
        self.p = [0.0 for x in range(self.K)]
        self.nw = [ [0 for y in range(self.K)] for x in range(self.dset.V) ]
        self.nwsum = [ 0 for x in range(self.K)]
        self.nd = [ [ 0 for y in range(self.K)] for x in range(self.dset.M)]
        self.ndsum = [ 0 for x in range(self.dset.M)]
        self.Z = [ [] for x in range(self.dset.M)]
        for x in range(self.dset.M):
            self.Z[x] = [0 for y in range(self.dset.docs[x].length)]
            self.ndsum[x] = self.dset.docs[x].length
            for y in range(self.dset.docs[x].length):
                topic = random.randint(0, self.K-1)
                self.Z[x][y] = topic
                self.nw[self.dset.docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1
        self.theta = [ [0.0 for y in range(self.K)] for x in range(self.dset.M) ]
        self.phi = [ [ 0.0 for y in range(self.dset.V) ] for x in range(self.K)]

    def estimate(self):
        #print ('Sampling %d iterations!' % self.iter_num)
        for x in range(self.iter_num):
            #print ('Iteration %d ...' % (x+1))
            for i in range(len(self.dset.docs)):
                for j in range(self.dset.docs[i].length):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
        #print ('End sampling.')
        #print ('Compute theta...')
        self.compute_theta()
        #print ('Compute phi...')
        self.compute_phi()
        #print ('Saving model...')
        self.save_model()

    def sampling(self, i, j):
        topic = self.Z[i][j]
        wid = self.dset.docs[i].words[j]
        self.nw[wid][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1

        Vbeta = self.dset.V * self.beta
        Kalpha = self.K * self.alpha

        for k in range(self.K):
            self.p[k] = (self.nw[wid][k] + self.beta)/(self.nwsum[k] + Vbeta) * \
                        (self.nd[i][k] + self.alpha)/(self.ndsum[i] + Kalpha)
        for k in range(1, self.K):
            self.p[k] += self.p[k-1]
        u = random.uniform(0, self.p[self.K-1])
        for topic in range(self.K):
            if self.p[topic]>u:
                break
        self.nw[wid][topic] += 1
        self.nwsum[topic] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1
        return topic

    def compute_theta(self):
        for x in range(self.dset.M):
            for y in range(self.K):
                self.theta[x][y] = (self.nd[x][y] + self.alpha) \
                                   /(self.ndsum[x] + self.K * self.alpha)

    def compute_phi(self):
        for x in range(self.K):
            for y in range(self.dset.V):
                self.phi[x][y] = (self.nw[y][x] + self.beta)\
                                 /(self.nwsum[x] + self.dset.V * self.beta)

    def save_model(self):
        with open(self.modelfile_suffix+'.theta', 'w') as ftheta:
            for x in range(self.dset.M):
                for y in range(self.K):
                    ftheta.write(str(self.theta[x][y]) + ' ')
                ftheta.write('\n')
        with open(self.modelfile_suffix+'.phi', 'w') as fphi:
            for x in range(self.K):
                for y in range(self.dset.V):
                    fphi.write(str(self.phi[x][y]) + ' ')
                fphi.write('\n')
        with open(self.modelfile_suffix+'.twords','w') as ftwords:
            if self.top_words > self.dset.V:
                self.top_words = self.dset.V
            for x in range(self.K):
                ftwords.write('Topic '+str(x)+'th:\n')
                topic_words = []
                for y in range(self.dset.V):
                    topic_words.append((y, self.phi[x][y]))
                #quick-sort
                topic_words.sort(key=lambda x:x[1], reverse=True)
                for y in range(self.top_words):
                    word = self.dset.id2word[topic_words[y][0]]
                    ftwords.write('\t'+word+'\t'+str(topic_words[y][1])+'\n')
        with open(self.modelfile_suffix+'.tassign','w') as ftassign:
            for x in range(self.dset.M):
                for y in \
                        range(self.dset.docs[x].length):
                    ftassign.write(str(self.dset.docs[x].words[y])+':'+str(self.Z[x][y])+' ')
                ftassign.write('\n')
        with open(self.modelfile_suffix+'.others','w') as fothers:
            fothers.write('alpha = '+str(self.alpha)+'\n')
            fothers.write('beta = '+str(self.beta)+'\n')
            fothers.write('ntopics = '+str(self.K)+'\n')
            fothers.write('ndocs = '+str(self.dset.M)+'\n')
            fothers.write('nwords = '+str(self.dset.V)+'\n')
            fothers.write('liter = '+str(self.iter_num)+'\n')
