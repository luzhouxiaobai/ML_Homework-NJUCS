import dataset
import lda

def main():
    data = "dataset/new1s.txt"
    wordbag = "dataset/wordbag.txt"
    dset = dataset.readtrnfile(data)
    model = lda.LDAModel(dset, K=10, alpha=0.1, beta=0.1, iter_num=10,
                         top_words=10, word_bags=wordbag, testdata=data,
                         finaltopics="dataset/final")
    model.init_est()
    model.estimate()

if __name__ == "__main__":
    main()