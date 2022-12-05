#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:22:30 2022

@author: davidteunissen
"""
To do
    Omschrijven naar eigen werkende functie
    Verander size naar resolution

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
    
    #All unique words
    KeyTitles = []
    for i in range(len(data)):
        for j in range(len(titles[i])):
            if titles[i][j] not in KeyTitles:
                KeyTitles.append(titles[i][j])
    
    #not scaleable?
    KeyTitles.remove('best'); KeyTitles.remove('buy')
    
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
    
#dat = cleandata(df); data = dat[0]; KeyTitles = dat[1]

#%%
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
    
#matrices = binSigM(data, KeyTitles, 1, 100); inputMatrix = matrices[0]; sigmatrix = matrices[1]; bandsFinal = matrices[2]
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

#candidates = candMat(bandsFinal, 100)
#%% Dissimilarity matrix
def disMatrix(candidate, inputmatrix, data):
    
    brand = data['brand']
    shop = data['shop']
    resol = data['resolution']

    def cosine_distance(A,B):
        distanceAB = 1 - np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return distanceAB
    
    dissimmatrix = np.ones((len(candidate),len(candidate))) * sys.maxsize 
    
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            if (candidate[i,j] == 1 and brand[i]== brand[j] and shop[i] != shop[j] and resol[i]==resol[j]):
                dist = cosine_distance(inputmatrix[:,i], inputmatrix[:,j])
                dissimmatrix[i,j] = dist
                dissimmatrix[j,i] = dist
                
    
    return dissimmatrix

#dissimMatrix = disMatrix(candidates, inputMatrix, data)

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

#dupes = clustering(dissimMatrix, 0.5, data); predDupes = dupes[0]; realDupes = dupes[1]

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

#perf(predDupes, realDupes, candidates, data)

#%% Dupe Predictor function

def dupePredictor(df, b, r, t):
    
    #Cleaning and splitting data
    dat = cleandata(df)
    data = dat[0]; KeyTitles = dat[1]
    
    # Create binary and signature matrix
    matrices = binSigM(data, KeyTitles, r, b)

    inputMatrix = matrices[0]; sigmatrix = matrices[1]; bandsFinal = matrices[2]
    
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


#results = dupePredictor(df, 100, 1, 0.50)
  
#%% Bootstrap
def booteval(df, nBoot, b, r, tresh):
    
    PQ = np.zeros(nBoot)
    PC = np.zeros(nBoot)
    
    F1 = np.zeros(nBoot)
    F1_Star = np.zeros(nBoot)
    
    comparisonfrac = np.zeros(nBoot)
    
    sizeBootdata = int(len(df)* 0.63)
    for i in range(nBoot):
        bootData = resample(df, n_samples= sizeBootdata, random_state=i)
        bootData = bootData.reset_index().iloc[:,1:6]
        eval = dupePredictor(bootData, b, r, tresh)
        
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

#results = booteval(df, 5, 100, 1, 0.50)


#%% Tuning parameters: threshhold for clustering
def tune(df, nBoot, b, r, treshvec):
    PQ = np.zeros(len(treshvec))
    PC = np.zeros(len(treshvec))
    F1 = np.zeros(len(treshvec))
    F1_Star = np.zeros(len(treshvec))
    comparisfrac = np.zeros(len(treshvec))
    
    for i in range(len(treshvec)):
        result = booteval(df, nBoot, b, r, treshvec[i])
        
        F1[i] = np.mean(result[0])
        F1_Star[i] = np.mean(result[1])
        PQ[i] = np.mean(result[2])
        PC[i] = np.mean(result[3])
        comparisfrac[i] = np.mean(result[4])
    
    return F1, F1_Star, PQ, PC, comparisonfrac
    
tvec = np.r_[0.2:1:0.1]

tune_results = tune(df, 5, 100, 1, tvec)

#Seemingly 0.5 treshhold workds best.

#%%
def tune_rb(df,nboot,tresh, br):
    PQ = np.zeros(len(br))
    PC = np.zeros(len(br))
    F1 = np.zeros(len(br))
    F1_Star = np.zeros(len(br))
    comparisfrac = np.zeros(len(br))

    for i in range(len(br)):
        result = booteval(df, nBoot, br[i][0], br[i][1], tresh)
        
        F1[i] = np.mean(result[0])
        F1_Star[i] = np.mean(result[1])
        PQ[i] = np.mean(result[2])
        PC[i] = np.mean(result[3])
        comparisfrac[i] = np.mean(result[4])
    
    return F1, F1_Star, PQ, PC, comparisonfrac


br = [(100,1), (20,5), (10,10), (100,5)]
tune_br_results = tune_rb(df, 5, 0.5, br)
tune_br_results

br2 = [(100,10),(100,1), (200,5)]
tune_br2_results = tune_rb(df, 5, 0.5, br2)
tune_br2_results

br3 = [(100,1), (700,1),(700,5)]
tune_br3_results = tune_rb(df, 5, 0.5, br3)
tune_br3_results

br4 = [(100,1), (700,5), (1000,10)]
tune_br4_results = tune_rb(df, 5, 0.5, br4)
tune_br4_results

res = dupePredictor(df, 700, 5, 0.8)
res

#%% Graphs

# Pairs for increasing fraction of comparisons - old
Frac = [0.8669738863287251, 0.6047487054115424, 0.5046302938481462, 0.3482023058691198, 0.31262489167924434, 0.27978261243837915, 0.1652126338825411, 0.08327795345029106]

F1 = [0.12702906322892354, 0.12494509422928282, 0.14316412743306417, 0.12746329925059485, 0.07004484681532269, 0.07075096096899784, 0.07364793862615207, 0.09389149277990601]
F1S = [0.00023648987610702318, 0.00033568827565493735, 0.00042358713098640717, 0.000488923446075783, 0.0002723885773414611, 0.0003572707118634843, 0.0008162816817441776, 0.0019921639215476723]
PQ = [0.00011843437838808116, 0.00016822278136378153, 0.00021223162069161262, 0.000245150890039713, 0.00013663432293557523, 0.0001793491459873674, 0.00041323879801915343, 0.001015126369519005]
PC = [0.09569393961334002, 0.09065669889730785, 0.10578730808263719, 0.0885914874558685, 0.0438166683936293, 0.04755274507773825, 0.04883436254004926, 0.06203748656233552]

# Pairs for increasing fraction of comparisons - new
Frac = [1.0, 0.8570900597414075, 0.7157927357662223, 0.5907011533171498, 0.346288973951369, 0.29554454972042243, 0.14742182254334266,0.10780885045136039]

F1 = [0.1290110467262165, 0.11854160074865099, 0.109548943377147,0.1294668421707212,0.11131892909959103, 0.07840382864653737,0.0734687575073227,0.07275994747781693]
F1S = [0.00018499951803920025, 0.00021470382013724366,0.00023315910886412776,0.0003167542694444707,0.0004355135667016458,0.0003239003025618535,0.000732087380495537,0.0011964666156506318]
PQ = [9.258674746964628e-05, 0.00010753995007952973, 0.00011682509066540243,0.00015864084821545663,0.00021838854099766967,0.00016247272455822905,0.00036902613127734257,0.0006082330327554935]
PC = [0.09867802518648151, 0.08865875736374548,  0.08039788779852809,0.0982479176595998, 0.07896328624791424, 0.05270806613484471, 0.048868972048914223,0.046963522868814625]

high3 = booteval(df, 5, 100, 1, 0.50) #1.0
high1 = booteval(df, 5, 25, 1, 0.50) #0.8570900597414075
temp1 = booteval(df, 5, 50, 1, 0.5) #0.7157927357662223


high4 = booteval(df, 5, 400, 2, 0.50) #0.5641432952082532
0.5907011533171498

high2 = booteval(df, 5, 200, 2, 0.50) #0.346288973951369
med2 = booteval(df, 5, 50, 2, 0.5) # 0.29554454972042243


#low1 = booteval(df, 5, 20, 5, 0.5) #0.29515966431565194

med1 = booteval(df, 5, 100, 6, 0.5) #0.14742182254334266
low2 = booteval(df, 5, 100, 7, 0.5) #0.10780885045136039

np.mean(low2[0]), np.mean(low2[1]), np.mean(low2[2]), np.mean(low2[3])



## Pair Quality
# Need: LSH results with addition (resolution) and without (only shops)
# Fraction of comparisons under both

## Pair Quality
plt.plot(Frac, PQ, label='Shop, Brand, Resolution');
plt.plot(Frac_brand, PQ_brand, label='Shop, Brand');
plt.plot(Frac_shops, PQ_shops, label='Shop');
plt.title('Pair Quality'); plt.xlabel('Fraction of Comparisons'); plt.ylabel('Pair Quality'); plt.legend();
plt.grid(); plt.show()

## Pair Completeness
plt.figure()
plt.plot(Frac, PC, label='Shop, Brand, Resolution');
plt.plot(Frac_brand, PC_brand, label='Shop, Brand');
plt.plot(Frac_shops, PC_shops, label='Shop');
plt.title('Pair Completeness'); 
plt.xlabel('Fraction of Comparisons'); plt.ylabel('Pair Completeness'); plt.legend(); plt.grid(); plt.show()

## F1
plt.figure()
plt.plot(Frac, F1, label='Shop, Brand, Resolution');
plt.plot(Frac_brand, F1_brand, label='Shop, Brand');
plt.plot(Frac_shops, F1_shops, label='Shop');
plt.title('F1')
plt.xlabel('Fraction of Comparisons'); plt.ylabel('F1'); plt.legend(); plt.grid(); plt.show()

## F1*
plt.figure()
plt.plot(Frac, F1S, label='Shop, Brand, Resolution');
plt.plot(Frac_brand, F1S_brand, label='Shop, Brand');
plt.plot(Frac_shops, F1S_shops, label='Shop');
plt.title('F1*')
plt.xlabel('Fraction of Comparisons'); plt.ylabel('F1'); plt.legend(); plt.grid(); plt.show()











