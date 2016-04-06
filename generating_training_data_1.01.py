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


def assemble_output_vector(anotacao):
    outvec=np.array(0)
    
    outvec = [0] * 201
    coord=np.zeros((2, 2))
    sa=len(anotacao)
    
    if sa>1:

        dist = [0] * sa
        centerx = [0] * sa
        centery = [0] * sa
        area = [0] * sa
        indexes = [0] * 2
        #getting the coordinates of the center of the object's bounding boxes:
        for j in range(sa):
            anota=anotacao[j]
            bb=np.array(anota['bbox'])
            centerx[j]=bb[0]+bb[2]/2
            centery[j]=bb[1]+bb[3]/2
            area[j]=bb[3]*bb[2]
        
        #getting the centroid of the set of objects:
        centroidx = np.mean(centerx)
        centroidy = np.mean(centery)
        
        #sorting objects from the most centered to the least centered
        for j in range(sa):
                dist[j] = (centerx[j] - centroidx)**2 - 0.1* area[j]
        sorted_indexes = sorted(range(len(dist)), key=lambda k: dist[k])
        indexes[0] = sorted_indexes[0]
        
        #getting the index of the most centered object:
        anota = anotacao[sorted_indexes[0]]
        category_obj_1 = anota['category_id']
        
        #searching for the second most centered object belonging to another category:
        for j in range(sa):
            anota = anotacao[sorted_indexes[j]]
            category_obj = anota['category_id']
            
            
            if category_obj != category_obj_1:
                indexes[1] = sorted_indexes[j]
                break
            
        coord[1][0] = centery[indexes[0]]
        coord[1][1] = centery[indexes[1]]
        coord[0][0] = centerx[indexes[0]]
        coord[0][1] = centerx[indexes[1]]
        
        #if the second object isn't the highest, flip the order of the objects:
        if coord[1][1] < coord[1][0]:
            aux = coord[1][0]
            coord[1][0] = coord[1][1]
            coord[1][1] = aux
            aux = coord[0][0]
            coord[0][0] = coord[0][1]
            coord[0][1] = aux
            aux = indexes[0]
            indexes[0] = indexes[1]
            indexes[1] = aux
        
        #calculating the gradient between the positions of the objects:    
        deltax = coord[0][1] - coord[0][0]
        deltay = coord[1][1] - coord[1][0]
        grad = deltay/deltax
        
        #annotating the relative position between objects:
            
        if (grad>1) or (grad<-1):
            outvec[0] = 1 #below(A,B)
            print 'below(A,B)'
        else:
            outvec[1] = 1 #besides(A,B)
            print 'besides(A,B)'
        
        #annotating the indexes of both visual objects (in one-hot style):
        anota=anotacao[indexes[0]]
        an0=anota['category_id']
        outvec[2+an0] = 1
        print an0
        if indexes[1] > 0:
            anota=anotacao[indexes[1]]
            an1=anota['category_id']
            outvec[100+an1] = 1
            print an1
        else:
            outvec[200] = 1 #the position 200 is for the absence of the second object 
            outvec[0] = 0   
            outvec[1] = 0
        
    
    else:
        if sa>0:       
            anota=anotacao[0]
            outvec[2+anota['category_id']] = 1        
            
     
    
    return outvec


# initializing COCO api for instance annotations
coco=COCO(annFile)

# get the image Ids:
imgIds = coco.getImgIds();

#creating the target output for all the captions:
s=len(imgIds)
out = np.zeros(shape=(5*s,201))
count=0
for imgId in imgIds:
    #getting the Id of the annotation corresponding to the image Id:
    annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
    #getting the annotation:
    anotacao=coco.loadAnns(ids=annIds)
    #creating the same target output for 5 captions:
    for k in range(5):
        out[count,:]=assemble_output_vector(anotacao)
        count += 1
    


# Saving the objects:
with open('training_target', 'w') as f:
    pickle.dump(out, f)



# If you want to get back the annotation:
#with open('training_target') as f:
#    out = pickle.load(f)


# initializing COCO api for caption annotations
annFile = 'captions_%s.json'%(dataType)
caps=COCO(annFile)


with open('training_sentences.csv', 'wb') as f:
        wri = csv.writer(f, skipinitialspace=True)
        for imgId in imgIds:
            annId = caps.getAnnIds(imgIds=imgId)
            captions = coco.showAnns(caps.loadAnns(annId))
            for n in range(5):
                if len(captions[n])<3:
                    captions[n]='none'
                cap = captions[n].lower()    
                wri.writerow([cap])    
    

