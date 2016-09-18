# Visually informed embedding of word (VIEW)

DESCRIPTION:

The visually informed embedding of word (VIEW) is a continuous vector representation for a word extracted from a deep neural model trained using the Microsoft COCO data set to forecast the spatial arrangements between visual objects, given a textual description. The model is composed of a deep multilayer perceptron (MLP) stacked on the top of a Long Short Term Memory (LSTM) network, the latter being preceded by an embedding layer. The VIEW can be applied to transferring multimodal background knowledge to NLP algorithms, i.e. VIEW can be concatenated to word2vec embedding to improve the encoding of spatial background knowledge. WIEW was evaluated in Spatial Role Labeling (SpRL) algorithms (which recognize spatial relations between objects mentioned in the text) using the Task 3 of SemEval-2013 benchmark data set, SpaceEval. In case of publication using this software package cite:

@ARTICLE{2016arXiv160308474L,
author = {Oswaldo Ludwig and Xiao Liu and Parisa Kordjamshidi and Marie-Francine Moens
},
title = "{Deep Embedding for Spatial Role Labeling}",
journal = {ArXiv e-prints},
archivePrefix = "arXiv",
eprint = {1603.08474},
primaryClass = "cs.CL",
keywords = {Computer Science - Computation and Language, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Learning, Computer Science - Neural and Evolutionary Computing},
year = 2016,
month = mar,
adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160308474L},
}

INSTALLATION AND USE:

1) Download the files “instances_val2014.json” and “captions_val2014.json” from http://mscoco.org/dataset/#download (the image files are not required, only these annotation files containing the position of the bounding boxes and captions);

2) Download in the same folder all of VIEW files, including the MS-COCO files included in this folder, since some of them were slightly modified to meet the VIEW requirements;

3) Run “generating_training_data_1.01.py” or “generating_training_data_1.02.py” to automatically derive the VIEW-style training annotation from the COCO annotation, generating the files “training_sentences.csv” and “training_target”. The codes “generating_training_data_1.01.py” and “generating_training_data_1.02.py” provide different annotation styles (the second yielded a better performance in recent experiments). It is possible to concatenate one or both visually informed embeddings (resulting from the different annotation styles) directly with word2vec. Optionally, you can first concatenate both visually informed embeddings, applying feature selection on them before the concatenation with word2vec, to avoid redundancy, see, for instance:
https://www.mathworks.com/matlabcentral/fileexchange/29553-feature-selector-based-on-genetic-algorithms-and-information-theory) ;

4) Run “training_deep_model_VIEW_1.01.py” to train the deep model from which VIEW is extracted (optionally you can also extract from this model the sentence-level embedding). This code generates the file “vocabulary”, containing the 4999 most frequent words in MS-COCO, and  the files “my_model_struct.json”, and “my_model_weights.h5”, with the model structure and weights, respectively. The libraries Theano and Keras are required, besides other libraries to NLP (see the import commands in the code). To run in GPU you can call the code like this: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python training_deep_model_VIEW_1.01.py

5) After training it is possible to see a plotting of the PCA projection of the resulting VIEW of some words by running “PCA_projection_1.01.py” (or “PCA_projection_1.02.py” for those who are using the latest version of Keras).

6) Run “saving_embedding_1.01.py” (or “saving_embedding_1.02.py” for those who are using the latest version of Keras) to save the VIEW vectors in the file “embedding_for_vocabulary” corresponding to the file “vocabulary”, which contains the 4999 most frequent words in MS-COCO. 

I also included Excel files containing the VIEW for those who don't want to run the Python code. "vocabulary_excel.xls" contains a vocabulary with the 4999 most frequent words of COCO and "embeddings_excel.xlsx" contains the respective visually informed embeddings of 200 features.
