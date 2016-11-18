import argparse
from functools import lru_cache
from logging import getLogger

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib


logger = getLogger(__name__)


class Word2VecCluster(Object):

    def __init__(self, model):
        if type(model) == str:
            self.classifier = joblib.load(model)
        else:
            self.classifier = model

    @lru_cache(maxsize=None)
    def to_class(self, vec):
        return self.classifier.predict([vec])


def make_dataset(model):
    """
    """
    V = model.index2word
    X = np.zeros((len(V), model.vector_size))

    for index, word in enumerate(V):
        X[index, :] += model[word]
    return X


def train(X, K):
    logger.info('start to fiting KMeans with {} classs.'.format(K))
    classifier = MiniBatchKMeans(n_clusters=K, random_state=0)
    classifier.fit(X)
    return classifier


def main():
    parser = argparse.ArgumentParser(
        description='Python Word2Vec Cluster')

    parser.add_argument('model',
                        action='store',
                        help='Name of word2vec binary modelfile.')

    parser.add_argument('-o', '--out',
                        action='store',
                        default='model.pkl',
                        help='Set output filename.')

    parser.add_argument('-k', '--K',
                        action='store',
                        type=int,
                        default=500,
                        help='Num of classes on KMeans.')

    parser.add_argument('-p', '--pre-trained-model',
                        action='store',
                        default=None,
                        help='Use pre-trained KMeans Model.')

    parser.add_argument('-w', '--words-to-pred',
                        action='store',
                        nargs='+',
                        type=str,
                        default=None,
                        help='List of word to predict.')

    args = parser.parse_args()

    model = Word2Vec.load_word2vec_format(args.model, binary=True)

    if not args.pre_trained_model:
        X = make_dataset(model)
        classifier = train(X, args.K)
        joblib.dump(classifier, args.out)
    else:
        classifier = joblib.load(args.pre_trained_model)

    if args.words_to_pred:

        X = [model[word] for word in args.words_to_pred if word in model]
        classes = classifier.predict(X)

        result = []
        i = 0
        for word in args.words_to_pred:
            if word in model:
                result.append(str(classes[i]))
                i += 1
            else:
                result.append(str(-1))
        print(' '.join(result))


if __name__ == '__main__':
    main()
