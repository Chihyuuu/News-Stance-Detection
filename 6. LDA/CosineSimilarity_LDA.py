from math import *
import json 
import pandas as pd

def cosine_similarity(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    if(dena != 0 and denb != 0):
        return numerator/sqrt(dena*denb)

# ------------------- train ------------------------
lda_train = pd.read_csv('X_lda_train.csv') 
# print(lda_train) # (150770, 11)
lda_train = lda_train.drop(columns=['index'])
# print(lda_train) # (150770, 10)
lda_train = lda_train.to_dict(orient="index")
# print (lda_train)

# print (cosine_similarity(lda_train[0],lda_train[49972]))

LDA_CosSim_train = {}
for i in range(0, 49972):
    LDA_CosSim_train.update({i:cosine_similarity(lda_train[i],lda_train[i+49972])})
    if LDA_CosSim_train[i] == None:
        LDA_CosSim_train[i] = 0.0
# print (LDA_CosSim_train) #(49972,1)

with open('./txt_CosineSimilarity/LDA_CS_train.txt','w') as file:
    file.write(json.dumps(LDA_CosSim_train))

# ------------------- test ------------------------
lda_test = pd.read_csv('X_lda_test.csv') 
# print(lda_test) # (50826, 11)
lda_test = lda_test.drop(columns=['index'])
# print(lda_test) # (50826, 10)
lda_test = lda_test.to_dict(orient="index")
# print (lda_test)

# print (cosine_similarity(lda_test[0],lda_test[49972]))

LDA_CosSim_test = {}
for i in range(0, 25413):
    LDA_CosSim_test.update({i:cosine_similarity(lda_test[i],lda_test[i+25413])})
    if LDA_CosSim_test[i] == None:
        LDA_CosSim_test[i] = 0.0
# print (LDA_CosSim_test) #(25413,1)

with open('./txt_CosineSimilarity/LDA_CS_test.txt','w') as file:
    file.write(json.dumps(LDA_CosSim_test))

