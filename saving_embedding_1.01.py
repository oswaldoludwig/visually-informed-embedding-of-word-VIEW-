__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

#This code saves the VIEW vectors, corresponding to the file "vocabulary" containing the 5000 most frequent words in MS-COCO, in the file "embedding_for_vocabulary".
#In case of publication, cite:   

##@ARTICLE{2016arXiv160308474L,
##   author = {{Ludwig}, O. and {Liu}, X. and {Kordjamshidi}, P. and {Moens}, M.-F.
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


import csv
import sys
import numpy as np
import mask
import itertools
import pickle
import theano
from keras.models import model_from_json
import nltk
import itertools
from keras.preprocessing import sequence
import time
from matplotlib.mlab import PCA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import mask


def get_embeddings(vocab,model_file,weights_file):
    unknown_token = "UNKNOWN_TOKEN"
    
    maxlen = len(vocab)
    
    print('The vocabulary has %s words.'%maxlen)
    #print vocab[10][0]
    #loading the trained VIEW model:
    
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    
    voc = ''    
    for n in range(maxlen/100):
        wo = str(vocab[n][0])
        #print type(wo)
        voc ='%s %s'%(voc,wo)
    
    print voc[0:50]
    name_list=[voc,voc]    
    print type(name_list)    
    unwanted_chars = ".,-_[]'"   
    clean_list = name_list
    cont=0
    for raw_word in name_list:
        clean_list[cont] = raw_word.strip(unwanted_chars)
        cont+=1

    # Tokenize the sentences into words
        
    tokenized_list = [nltk.word_tokenize(sents) for sents in clean_list]    
    

    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    for i, sents in enumerate(tokenized_list):
        tokenized_list[i] = [w if w in word_to_index else unknown_token for w in sents]


    #Assembling the input vector, i.e. substituting words by indexes:
    X = np.asarray([[word_to_index[w] for w in string] for string in tokenized_list])
    
    print("Pad sequences (samples x time)")
    X = sequence.pad_sequences(X, maxlen=maxlen)

    X = map(list,X)

    #defining a Theano function to get the embedding:
    get_embedding = theano.function([model.layers[0].input], model.layers[0].get_output(train=False))    
    print('I have the Theano function')
    #Instantiating X to have the embedding:
    embeddings=get_embedding(X)
    embeddings=embeddings[0]
    a, b = embeddings.shape
    print('The corresponding embedding has dimension %s x %s'%(a,b))
    
    return embeddings


weights_file='my_model_weights.h5'
model_file='my_model_struct.json'
with open('vocabulary', 'r') as v:
        vocab=pickle.load(v)


embeddings = get_embeddings(vocab,model_file,weights_file)

with open('embedding_for_vocabulary', 'w') as f:
    pickle.dump(embeddings, f)

