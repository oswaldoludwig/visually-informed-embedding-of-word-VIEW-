__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

#This code plots the PCA projection of VIEW vectors corresponding to some words (see line 43).
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


#***************************************************************************************
#write here the words (until 30 words) that you want to visualise in the PCA projection:
#***************************************************************************************

name_list='cake sandwich bottle spoon truck car train bicycle motorbike man boy girl bird dog glass bus pizza'

maxlen = 17 # set here the number of words of visual objects



#name_list='above top below bottom on over under down'

#maxlen = 8 # set here the number of words of spatial relation indicators
#***************************************************************************************



def projection(embeddings, token_list):
    for k in range(6):
        embeddings=np.concatenate((embeddings, embeddings), axis=0)
    proj = PCA(embeddings)
    PCA_proj=proj.Y
    print PCA_proj.shape
    
    #plotting words within the 2D space of the two principal components:
    list=token_list[0]
    
        
    for n in range(maxlen):
        plt.plot(PCA_proj[n][0]+1,PCA_proj[n][1], 'w.')
        plt.annotate(list[n], xy=(PCA_proj[n][0],PCA_proj[n][1]), xytext=(PCA_proj[n][0],PCA_proj[n][1]))
    plt.show()       
    plt.ishold()
    
  
    return

def get_embeddings(name_list,model_file,weights_file):
    unknown_token = "UNKNOWN_TOKEN"
    
    
    #loading the trained VIEW model:
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    # Loading vocabulary:
    with open('vocabulary', 'r') as v:
        vocab=pickle.load(v)
        
    name_list=[name_list,name_list]    
        
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
    get_embedding = theano.function([model.layers[0].input], model.layers[0].output)
    print('I have the Theano function')
    #Instantiating X to have the embedding:
    embeddings=get_embedding(X)
    embeddings=embeddings[0]
    
    return embeddings, tokenized_list



#weights_file='my_model_weights_2.h5'
weights_file='my_model_weights.h5'
model_file='my_model_struct.json'

embeddings, tokenized_list = get_embeddings(name_list,model_file,weights_file)
projection(embeddings, tokenized_list)
