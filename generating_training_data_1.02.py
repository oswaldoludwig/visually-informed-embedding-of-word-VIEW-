__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

#This code automatically derive (from the COCO annotations) the VIEW-style training annotation: training_sentences.csv and training_target.
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


from coco import COCO
import csv
import sys
import numpy as np
import pylab
import mask
import itertools
import pickle
import time

dataDir='..'
dataType='val2014'
annFile='instances_%s.json'%(dataType)

def assemble_output_vector2(anotacao):
    outvec=np.array(0)
    flag = 0
    outvec = [0] * 201
    coord=np.zeros((2, 2))
    sa=len(anotacao)
  
    if sa>1:

        salience = [0] * sa
        indexes = [0] * 2
        #getting the coordinates of the two most salient objects:
        for j in range(sa):
            anota=anotacao[j]
            bb=np.array(anota['bbox'])
            salience[j]=bb[2]*bb[3]- 0.3*(320-(2*bb[0]+bb[2])/2)**2 #this is a emprical measure of salience of the objects given their area and position      
        
        for i in range(2):
            max_salience = 0
            for j in range(sa):
                if salience[j]>max_salience:
                    max_salience=salience[j]
                    indexes[i]=j
            salience[indexes[i]]=0
            anota=anotacao[indexes[i]]
            bb=np.array(anota['bbox'])
            
            coord[0][i]=(2*bb[0]+bb[2])/2 #coordinates of the center of the bounding box
            coord[1][i]=(2*bb[1]+bb[3])/2
                    
        if coord[1][1]<coord[1][0]: #flipping the objects, if the second isn't the highest
            aux =  coord[1][0]
            coord[1][0]=coord[1][1]
            coord[1][1]=aux
            aux =  coord[0][0]
            coord[0][0]=coord[0][1]
            coord[0][1]=aux
            flag = 1
            
        detax=coord[0][1]-coord[0][0]
        detay=coord[1][1]-coord[1][0]
        if detax == 0:
            grad = 0
        else:
            grad=detay/detax
        
        #defining the relative position between objects
            
        if (grad>1) or (grad<-1):
            outvec[0] = 1 #below(A,B)
        else:
            outvec[1] = 1 #besides(A,B)
        
        
        anota=anotacao[indexes[0]]
        an0=anota['category_id']
        anota=anotacao[indexes[1]]
        an1=anota['category_id']
        
        if flag == 0:
            outvec[2+an0] = 1
            outvec[100+an1] = 1
        else:
            outvec[2+an1] = 1
            outvec[100+an0] = 1
                
        
    
    else:
        if sa>0:       
            anota=anotacao[0]
            outvec[2+anota['category_id']] = 1        
            
    
    return outvec


# initialize COCO api for instance annotations
coco=COCO(annFile)

# get the image Ids:
imgIds = coco.getImgIds();
s=len(imgIds)
out = np.zeros(shape=(5*s,201))
count=0
for imgId in imgIds:
    
    #get the Id of the annotation corresponding to the image Id:
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    #get the annotation:
    anotacao=coco.loadAnns(ids=annIds)
   
    for k in range(5):
        out[count,:]=assemble_output_vector2(anotacao)
        count += 1

# Saving the objects:
with open('training_target', 'w') as f:
    pickle.dump(out, f)

# initialize COCO api for caption annotations
annFile = 'captions_%s.json'%(dataType)
caps=COCO(annFile)

with open('training_sentences.csv', 'wb') as f:
        wri = csv.writer(f, skipinitialspace=True)
        for imgId in imgIds:
            annId = caps.getAnnIds(imgIds=imgId)
            #print caps.loadAnns(annId)
            captions = coco.showAnns(caps.loadAnns(annId))
            #print captions
            for n in range(5):
                #print captions[n]
                if len(captions[n])<3:
                    
                    captions[n]='none'
                cap = captions[n].lower()    
                wri.writerow([cap])    
    

