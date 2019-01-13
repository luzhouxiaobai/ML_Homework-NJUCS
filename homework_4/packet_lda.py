from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import xlwt
def text_processing():
    text = open("dataset/new1s.txt", encoding = "UTF-8")
    text_train = [tok for tok in text if len(tok)>1]
    vect = CountVectorizer(min_df = 2, stop_words = "english")#.fit(text_train)#创建词表，去除无用词
    print(type(vect))
    print(vect.vocabulary_)
    text_final = vect.fit_transform(text_train)
    print(text_final)
    return vect, text_final

def main(vect, text, k):
    lda = LatentDirichletAllocation(n_components = k, learning_method = "batch")
    document_topics = lda.fit_transform(text)
    sorting = np.argsort(lda.components_, axis = 1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    (a, b)=lda.components_.shape
    sum = []
    for i in range(a):
        temp = 0
        for j in range(b):
            temp += lda.components_[i][j]
        sum.append(temp)
    #print(document_topics)

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    '''
    sheet = book.add_sheet('test', cell_overwrite_ok=True)
    sheet.write(0, 0, 'k = {}'.format(k))
    h=1
    l1=0
    l2=1
    '''

    for i in range(a):
        '''
        h=1
        sheet.write(h, l1, 'topic {}'.format(i))
        sheet.write(h, l2, '概率 {}'.format(i))
        '''
        print("topic{}: ".format(i))
        temp = sum[i]
        for j in range(10):
            #h=h+1
            k = sorting[i][j]
            '''
            sheet.write(h, l1, '{}'.format(feature_names[k]))
            sheet.write(h, l2, '%.2f%%'%(lda.components_[i][k]/temp*100))
            '''
            print("{}: {}".format(feature_names[k], lda.components_[i][k]/temp), end=" ")
        print("")
        '''
        l1=l1+2
        l2=l2+2
    book.save(r'e:\test.xls')
    '''

if __name__ == "__main__":
    a, b = text_processing()
    #main(a, b, 5)
    main(a, b, 10)
    #main(a, b, 20)