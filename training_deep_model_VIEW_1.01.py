__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

#This code trains the VIEW deep model, given the VIEW-style training annotation, and outputs my_model_struct.json, and my_model_weights.h5.
#In case of publication, cite:

##@ARTICLE{2016arXiv160308474L,
##   author = {Oswaldo Ludwig and Xiao Liu and Parisa Kordjamshidi and Marie-Francine Moens
##	},
##    title = "{Deep Embedding for Spatial Role Labeling}",
##  journal = {ArXiv e-prints},
##archivePrefix = "arXiv",
##   eprint = {1603.08474},
## primaryClass = "cs.CL",
## keywords = {Computer Science - Computation and Language, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Learning, Computer Science - Neural and Evolutionary Computing},
##     year = 2016,
##    month = mar,
##   adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160308474L},
##}


import numpy as np
np.random.seed(1237)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.regularizers import ActivityRegularizer
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
from keras.models import model_from_json
import os
import csv
import sys
import nltk
import itertools
import operator
import pickle
import theano
import theano.tensor as T

# To run in GPU:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python training_deep_model_VIEW_1.01.py


vocabulary_size = 5000 #_VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


max_features = vocabulary_size
maxlen = 30  # cut texts after this number of words (among top max_features most common words)
batch_size = 1000

#Our custom objective function:
def hinge2(y_true, y_pred):
    return T.mean((T.mean(T.maximum((1. - (2. * y_true - 1.) * y_pred), 0.), axis=1)))


# Getting the target:
with open('training_target') as f:
    y = pickle.load(f)

# Reading the data:
print ("Reading CSV file...")
with open('training_sentences.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    sentences = reader
    sentences = ["%s" % x for x in sentences]
    
print ("Parsed %d sentences." % (len(sentences)))


unwanted_chars = ".,-_[]'"

clean_sentences = sentences
cont=0
for raw_word in sentences:
    clean_sentences[cont] = raw_word.strip(unwanted_chars)
    cont+=1

# Tokenizing the sentences into words:
tokenized_sentences = [nltk.word_tokenize(sent) for sent in clean_sentences]


# Counting the word frequencies:
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

# Getting the most common words and build index_to_word and word_to_index vectors:
vocab = word_freq.most_common(vocabulary_size-1)

# Saving vocabulary:
with open('vocabulary', 'w') as v:
    pickle.dump(vocab, v)

index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replacing all words not in our vocabulary with the unknown token:
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


# Creating the training data:
print (tokenized_sentences[0])
X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])

print (X.shape)
print (y.shape)

X = map(list,X)
y = map(list,y)

X_train=X[0:199520]
X_test=X[199521:202519]
y_train=y[0:199520]
y_test=y[199521:202519]

print(X_train[0])

print (len(y_train))
print (len(X_train))

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 200, input_length=maxlen))
model.add(LSTM(300)) 
model.add(Dense(210,init='he_normal',W_constraint = maxnorm(2)))
model.add(Activation('sigmoid'))
model.add(Dense(205,init='he_normal',W_constraint = maxnorm(2)))
model.add(Activation('tanh'))
model.add(Dense(201,init='he_normal',W_constraint = maxnorm(2)))
model.compile(loss=hinge2, optimizer='adam')

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=25, validation_data=(X_test, y_test))
json_string = model.to_json()
open('my_model_struct.json', 'w').write(json_string)
print('model structure saved')
model.save_weights('my_model_weights.h5')
print('model weights saved')


model.compile(loss='mean_squared_error', optimizer='adam')
json_string = model.to_json()
open('my_model_struct.json', 'w').write(json_string)
print('model structure saved again')











