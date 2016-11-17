# Python K-Means Clustring of Word2Vec


## How to Use

```
usage: main.py [-h] [-o OUT] [-k K] [-s SAMPLING_SIZE] [-p PRE_TRAINED_MODEL]
               [-w WORDS_TO_PRED [WORDS_TO_PRED ...]]
               model

Python Word2Vec Cluster

positional arguments:
  model                 Name of word2vec binary modelfile.

optional arguments:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Set output filename.
  -k K, --K K           Num of classes on KMeans.
  -s SAMPLING_SIZE, --sampling-size SAMPLING_SIZE
                        Num of sumpling size to use training dataset.
  -p PRE_TRAINED_MODEL, --pre-trained-model PRE_TRAINED_MODEL
                        Use pre-trained KMeans Model.
  -w WORDS_TO_PRED [WORDS_TO_PRED ...], --words-to-pred WORDS_TO_PRED [WORDS_TO_PRED ...]
                        List of word to predict.
```
