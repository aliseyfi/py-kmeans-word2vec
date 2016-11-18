# Python K-Means Cluster of Word2Vec

## Basic Usage

### Setup

You download Word2Vec model file such as [Google Code word2vec](https://code.google.com/archive/p/word2vec/).  
In this document, We use GoogleNews-vectors-negative300.bin.gz.  

### Train

```
$ python3 w2vcluster/w2vcluster.py GoogleNews-vectors-negative300.bin -k 500 -o model1000.pkl
```

### Predict

You can use command line interface.

```
$ python3 w2vcluster/w2vcluster.py GoogleNews-vectors-negative300.bin -p model500.pkl -w apple Apple banana Google
176 118 176 118
```

These integer values indicte cluster id of each words.  
Also you can use python interface.

```
$ python3
>>> from gensim.models.word2vec import Word2Vec
>>> from sklearn.externals import joblib
>>> model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
>>> classifier = joblib.load('model500.pkl')
>>> classifier.predict([model[w] for w in ['apple', 'Apple', 'banana', 'Google']])
array([176, 118, 176, 118], dtype=int32)
```

## How to Use

```
usage: w2vcluster.py [-h] [-o OUT] [-k K] [-p PRE_TRAINED_MODEL]
                     [-w WORDS_TO_PRED [WORDS_TO_PRED ...]]
                     model

Python Word2Vec Cluster

positional arguments:
  model                 Name of word2vec binary modelfile.

optional arguments:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Set output filename.
  -k K, --K K           Num of classes on KMeans.
  -p PRE_TRAINED_MODEL, --pre-trained-model PRE_TRAINED_MODEL
                        Use pre-trained KMeans Model.
  -w WORDS_TO_PRED [WORDS_TO_PRED ...], --words-to-pred WORDS_TO_PRED [WORDS_TO_PRED ...]
                        List of word to predict.
```
