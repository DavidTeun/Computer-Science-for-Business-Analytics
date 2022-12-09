#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:17:06 2022

@author: davidteunissen
"""

#%% Packages and wd
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
import time
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.cluster import AgglomerativeClustering
from scipy import spatial



# Setting working directory

os.chdir('/Users/davidteunissen/Desktop/Computer Science/Paper')
#%% Importing Data set
with open("TVs-all-merged.json") as jsonFile:
    Json = json.load(jsonFile)

## Next, we create a dataframe and split the modelID from the data set
temp = []
for i in range(len(list(Json.values()))):
    if len(list(Json.values())[i]) == 1:
        temp.append(list(Json.values())[i][0])
    else:
        for k in range(len(list(Json.values())[i])):
            temp.append(list(Json.values())[i][k])


df = pd.DataFrame(temp)
#%%
def cleandata(data):
    modelid_raw = (data['modelID'])
    title_raw = (data['title'])
    shop = (data['shop'])
    
    #Model ID
    ModelIDs = []

    for i in range(len(modelid_raw)):
      temp1 = re.sub(r'[^\w\s]','', modelid_raw[i])
      temp1 = temp1.lower()
      ModelIDs.append(temp1)
    
   #Titles
    for i in range(len(title_raw)):
        title_raw[i] = title_raw[i].replace('-', '')
        title_raw[i] = title_raw[i].replace('/', '')
        title_raw[i] = title_raw[i].replace(':', '')
        title_raw[i] = title_raw[i].replace('–', '')
        title_raw[i] = title_raw[i].replace(';','')
        title_raw[i] = title_raw[i].replace('+', '')
        title_raw[i] = title_raw[i].replace('(', '')
        title_raw[i] = title_raw[i].replace(')', '')
        title_raw[i] = title_raw[i].replace('[', '')
        title_raw[i] = title_raw[i].replace('.'," ")
        title_raw[i] = title_raw[i].replace(','," ")
        title_raw[i] = title_raw[i].replace('  '," ")
        title_raw[i] = title_raw[i].replace('"', 'inch')
        title_raw[i] = title_raw[i].replace('\"', 'inch')
        title_raw[i] = title_raw[i].replace("'"," ")
        title_raw[i] = title_raw[i].replace('inches', 'inch')
        title_raw[i] = title_raw[i].replace('-inch', 'inch')
        title_raw[i] = title_raw[i].replace(' inch', 'inch')
        title_raw[i] = title_raw[i].replace(' hz', 'hz')
        title_raw[i] = title_raw[i].replace('hertz', 'hz')
        title_raw[i] = title_raw[i].replace('Hertz', 'hz')
        
   
    titles = []
    for i in range(len(data)):
        temp2 = re.sub(r'[^\w\s]','', title_raw[i]) 
        temp2 = temp2.lower()
        temp2 = temp2.split()
       
        titles.append(temp2)
    
    KeyTitles = []
    for i in range(len(data)):
        for j in range(len(titles[i])):
            if titles[i][j] not in KeyTitles:
                KeyTitles.append(titles[i][j])
    
    KeyTitles.remove('best'); KeyTitles.remove('buy'); KeyTitles.remove('refurbished')
    
    brands =  ["philips", "supersonic", "sharp", "samsung", 
               "toshiba", "hisense", "sony", "lg",  "sanyo",
               "coby", "panasonic", "rca", "vizio", "naxa",
               "sansui", "viewsonic", "avue", "insignia",
               "sunbritetv", "magnavox", "jvc", "haier", 
               "optoma", "nec", "proscan", "venturer", 
               "westinghouse", "pyle", "dynex", "magnavox", 
               "sceptre", "tcl", "mitsubishi", 
               "curtisyoung", "compaq", "hannspree", 
               "upstar", "azend", "seiki", "craig",
               "contex", "affinity", "hiteker", "epson", 
               "elo", "pyle", "gpx", "sigmac", 
               "venturer", "elite"]
    
    tvBrand = np.zeros(len(titles))
    for i in range(len(titles)):
        for j in range(len(brands)):
            if brands[j] in titles[i]:
                tvBrand[i] = j
    
    resols = ["720p", "1080p", "4k"]
    
    tvResol = np.zeros(len(titles))
    for i in range(len(titles)):
        for j in range(len(resols)):
            if resols[j] in titles[i]:
                tvResol[i] = j
    
    datas =  pd.concat([pd.Series(ModelIDs), pd.Series(titles), shop, pd.Series(tvBrand), pd.Series(tvResol)], axis=1 )
    
    datas.set_axis([ 'modelID', 'title', 'shop', 'brand', 'resolution'], axis='columns', inplace=True) 

    return datas, KeyTitles
    
#%% Binary and Signature Matrix
def binSigM(data, keytitles, r, b):
    n = r*b
    p=9973
    titles = data['title']
    
    #Binary matrix
    def binMatrix(titles,Keytitles):
        inputMatrix = np.zeros((len(Keytitles), len(titles)))
        for k in range(len(Keytitles)):
            for i in range(len(titles)):
                if Keytitles[k] in titles[i]:
                    inputMatrix[k,i] = 1
        return inputMatrix
    
    inputMatrix = binMatrix(titles, keytitles)
    
    #Minhashing
    def randomIntVec(n):
        randomInt = []
        for i in range(n):
            r_int = random.randint(0,n)
            randomInt.append(r_int)
        return randomInt

    randomA = randomIntVec(n)
    randomB = randomIntVec(n)

    def hashfunc(x,a,b,p):
            return (a*x + b)%p

    def minhash(data, perm, randomIntA, randomIntB):
        rows, cols, sigrows = len(data), len(data[0]), perm
        
        sigmatrix = np.ones((n, len(titles))) * sys.maxsize
        for r in range(rows):
            hashvalue = []
            for k in range(perm):
                hashvalue.append(hashfunc(r,randomIntA[k],randomIntB[k],p))
            
            for c in range(cols):
                if data[r][c] == 0:
                    continue
                for i in range(sigrows):
                    if sigmatrix[i][c] > hashvalue[i]:
                        sigmatrix[i][c] = hashvalue[i]
    
        return sigmatrix
    
    sigmatrix = minhash(inputMatrix, n, randomA, randomB)
    
    z = np.arange(0, n, r)
    
    bandsDict_Vhash = {}
    for i in range(0,len(z)-1):
        bandsDict_Vhash["band{0}".format(i)]=sigmatrix[z[i]:z[i+1]][:]
    bandsDict_Vhash["band{0}".format(b-1)]=sigmatrix[z[b-1]:][:]

    bandsFinal = []
    for i in range(0,b):
        band_temp = []
        for j in range(len(sigmatrix[1,:])):
            bandtemp1 = bandsDict_Vhash['band'+str(i)][:,j] 
            bandtemp2 = [str(int) for int in bandtemp1]
            
            for k in range(len(bandtemp2)):
                bandtemp2[k] = bandtemp2[k].replace('.0','')
            
            bandtemp3 = ''.join(bandtemp2)
            band_temp.append(bandtemp3)
            
        bandsFinal.append(band_temp)
        
        
    return inputMatrix, sigmatrix, bandsFinal

#%% Candidate Matrix
def candMat(bandList, b):
    candidate = np.zeros((len(bandList[1]), len(bandList[1])))
    
    for i in range(len(bandList[0])):
        for j in range(i+1, len(bandList[0])):
            for c in range(0,b):
                if (bandList[c][i] == bandList[c][j]):
                    candidate[i,j] = 1
                    candidate[j,i] = 1
                    break
    return candidate

#%% Dissimilarity matrix
def disMatrix(candidate, inputmatrix, data):
    
    brand = data['brand']
    shop = data['shop']
    resol = data['resolution']

    def cosine_distance(A,B):
        distanceAB = 1 - np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return distanceAB
    
    dissimmatrix = np.ones((len(candidate),len(candidate))) * sys.maxsize 
    
    
    #To compare results, we change the preselection methods used here.
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            if (candidate[i,j] == 1 and brand[i]== brand[j] and shop[i] != shop[j] and resol[i]==resol[j]):
                dist = cosine_distance(inputmatrix[:,i], inputmatrix[:,j])
                dissimmatrix[i,j] = dist
                dissimmatrix[j,i] = dist
    
    for i in range(len(candidate)):
        for j in range(len(candidate)):
                if dissimmatrix[i,j] == 0:
                    dissimmatrix[i,j] = sys.maxsize
    return dissimmatrix

#%% Clustering
def clustering(dissimMatrix, t, data):
    cl = AgglomerativeClustering(affinity='precomputed', 
                                         linkage='complete', distance_threshold = t, n_clusters=None)
    cl = cl.fit(dissimMatrix)
    
    lab = cl.labels_
    predDupes = []
    
    for i in range(0, cl.n_clusters_):
        cp = np.where(lab == i)[0]
        if (len(cp)>1):
            predDupes.extend(list(combinations(cp, 2)))
            
    IDs = data['modelID']
    
    realDupes = []
    
    for modelID in IDs:
        if modelID not in realDupes:
            dp = np.where(IDs == modelID)[0]
            if (len(dp)>1):
                realDupes.extend(list(combinations(dp,2)))
    
    realDupes = list(set(realDupes))
    
    return predDupes, realDupes

#%% Final performance function
def perf(pred_dupes, real_dupes, candidate, data):
    
    nPredDupes = len(pred_dupes)
    nRealDupes = len(real_dupes)
    
    TP=[]; FP=[]
    for i in range(0,nPredDupes):
        if pred_dupes[i] in real_dupes:
            TP.append(pred_dupes[i])
        else:
            FP.append(pred_dupes[i])
    
    nTP = len(TP)
    nFP = len(FP)
    nFN = nRealDupes - len(TP)
    
    nComps = np.count_nonzero(candidate)/2
    nCompsPossible = len(data)*(len(data)-1) * 0.5
    
    comparisonFrac = nComps/nCompsPossible
    
    PQ = nTP/nComps
    PC = nTP/nRealDupes
    
    precision = nTP/(nTP+nFP)
    recall = nTP/(nTP + nFN)
    
    def F1(precision, recall):
        f1 = (2*precision*recall)/(precision+recall)
        return f1
    
    def F1Star(PQ, PC):
        f1star = (2*PQ*PC)/(PQ+PC)
        return f1star
    
    F1 = F1(precision, recall)
    F1Star = F1Star(PQ, PC)
    
    return nTP, PQ, PC, F1, F1Star, comparisonFrac


#%% Dupe Predictor function
def dupePredictor(df, b, r, t):
    
    #Cleaning and splitting data
    dat = cleandata(df)
    data = dat[0]; KeyTitles = dat[1]
    
    # Create binary and signature matrix
    matrices = binSigM(data, KeyTitles, r, b)

    inputMatrix = matrices[0]; bandsFinal = matrices[2]
    
    # Extract candidate and dissimilarity matrix 
    candidates = candMat(bandsFinal, b)
    
    dissimMatrix = disMatrix(candidates, inputMatrix, data)
    
    # Clustering, extracting real duplicates and predicted duplicates
    dupes = clustering(dissimMatrix, t, data)
    predDupes = dupes[0]; realDupes = dupes[1]
    
    # Performance measures
    performance = perf(predDupes, realDupes, candidates, data)
    
    print("Predicted duplicates:", len(predDupes))
    print("TP:", performance[0])
    print("PQ:", performance[1])
    print("PC:", performance[2])
    print("F1:", performance[3])
    print("F1 Star:", performance[4])
    print("Fraction of comparisons made", performance[5])

    return len(predDupes), performance
  
#%% Bootstrap
def booteval(df, nBoot, b, r, tresh):
    
    PQ = np.zeros(nBoot)
    PC = np.zeros(nBoot)
    
    F1 = np.zeros(nBoot)
    F1_Star = np.zeros(nBoot)
    
    comparisonfrac = np.zeros(nBoot)
    
    for i in range(nBoot):
        modelIDs = df['modelID']
        titles = df['title']
        shop = df['shop']
        
        df2 = pd.concat([pd.Series(modelIDs), pd.Series(titles), pd.Series(shop)],axis=1)
        
        bootData = resample(df2, n_samples= len(df2),random_state=i)
        
        train = bootData.drop_duplicates()
        
        test = pd.concat([df2, train]).drop_duplicates(keep=False)
        test = test.reset_index().iloc[:,1:4]
        eval = dupePredictor(test, b, r, tresh)
        
        PQ[i] = eval[1][1]
        PC[i] = eval[1][2]
        F1[i] = eval[1][3]
        F1_Star[i] = eval[1][4]
        comparisonfrac[i] = eval[1][5]
    
    print("F1: Avg:", np.mean(F1))
    print("F1 Star: Avg:", np.mean(F1_Star))
    print("PQ: Avg:", np.mean(PQ))
    print("PC: Avg:", np.mean(PC))
    print("Fraction: Avg:", np.mean(comparisonfrac))
    
    return F1, F1_Star, PQ, PC, comparisonfrac

booteval(df, 5, 800,1,0.5)

#%% Tuning parameters: threshhold for clustering
def tune(df, nBoot, b, r, treshvec):
    PQ = np.zeros(len(treshvec))
    PC = np.zeros(len(treshvec))
    F1 = np.zeros(len(treshvec))
    F1_Star = np.zeros(len(treshvec))
    comparisonfrac = np.zeros(len(treshvec))
    
    for i in range(len(treshvec)):
        modelIDs = df['modelID']
        titles = df['title']
        shop = df['shop']
        
        df2 = pd.concat([pd.Series(modelIDs), pd.Series(titles), pd.Series(shop)],axis=1)
        
        bootData = resample(df2, n_samples= len(df2),random_state=i)
        
        train = bootData.drop_duplicates()
        train = train.reset_index().iloc[:,1:4]
        result = booteval(train, nBoot, b, r, treshvec[i])
        
        F1[i] = np.mean(result[0])
        F1_Star[i] = np.mean(result[1])
        PQ[i] = np.mean(result[2])
        PC[i] = np.mean(result[3])
        comparisonfrac[i] = np.mean(result[4])
    
    return F1, F1_Star, PQ, PC, comparisonfrac
    

#%% Bootstrap results
high3 = booteval(df,5,800,1,0.50) 
high2 = booteval(df,5,400,2,0.50) 
Temp = booteval(df,5,267,3,0.50) 
high1 = booteval(df,5,200,4,0.50) 
med3 = booteval(df,5,160,5,0.50) 
med2 = booteval(df,5,134,6,0.50) 
med1 = booteval(df,5,114,7,0.50) 
low2 = booteval(df,5,100,8,0.50) 
low1 = booteval(df,5,89,9,0.50) 

Frac = [np.mean(high3[4]), np.mean(high2[4]), np.mean(Temp[4]), np.mean(high1[4]), np.mean(med3[4]), np.mean(med2[4]), np.mean(med1[4]), np.mean(low2[4]), np.mean(low1[4])]
              
F1 = [np.mean(high3[0]), np.mean(high2[0]), np.mean(Temp[0]), np.mean(high1[0]), np.mean(med3[0]), np.mean(med2[0]), np.mean(med1[0]), np.mean(low2[0]), np.mean(low1[0])]
F1S = [np.mean(high3[1]), np.mean(high2[1]), np.mean(Temp[1]), np.mean(high1[1]), np.mean(med3[1]), np.mean(med2[1]), np.mean(med1[1]), np.mean(low2[1]), np.mean(low1[1])]
PQ = [np.mean(high3[2]), np.mean(high2[2]), np.mean(Temp[2]), np.mean(high1[2]), np.mean(med3[2]), np.mean(med2[2]), np.mean(med1[2]), np.mean(low2[2]), np.mean(low1[2])]
PC= [np.mean(high3[3]), np.mean(high2[3]), np.mean(Temp[3]), np.mean(high1[3]), np.mean(med3[3]), np.mean(med2[3]), np.mean(med1[3]), np.mean(low2[3]), np.mean(low1[3])]

## Pair Quality
plt.plot(Frac, PQ, label='Shop, Brand, Resolution');
plt.title('Pair Quality'); plt.xlabel('Fraction of Comparisons'); plt.ylabel('Pair Quality'); plt.legend();
plt.grid(); plt.show()

## Pair Completeness
plt.figure()
plt.plot(Frac, PC, label='Shop, Brand, Resolution');
plt.title('Pair Completeness'); 
plt.xlabel('Fraction of Comparisons'); plt.ylabel('Pair Completeness'); plt.legend(); plt.grid(); plt.show()

## F1
plt.figure()
plt.plot(Frac, F1, label='Shop, Brand, Resolution');
plt.title('F1')
plt.xlabel('Fraction of Comparisons'); plt.ylabel('F1'); plt.legend(); plt.grid(); plt.show()

## F1*
plt.figure()
plt.plot(Frac, F1S, label='Shop, Brand, Resolution');
plt.title('F1*')
plt.xlabel('Fraction of Comparisons'); plt.ylabel('F1*'); plt.legend(); plt.grid(); plt.show()


