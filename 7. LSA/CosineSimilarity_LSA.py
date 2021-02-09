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
lsa_train = pd.read_csv('X_lsa_train.csv') 
# print(lsa_train) # (150770, 11)
lsa_train = lsa_train.drop(columns=['index'])
# print(lsa_train) # (150770, 10)
lsa_train = lsa_train.to_dict(orient="index")
# print (lsa_train)

# print (cosine_similarity(lsa_train[0],lsa_train[49972]))

LSA_CosSim_train = {}
for i in range(0, 49972):
    LSA_CosSim_train.update({i:cosine_similarity(lsa_train[i],lsa_train[i+49972])})
    if LSA_CosSim_train[i] == None:
        LSA_CosSim_train[i] = 0.0
# print (LSA_CosSim_train) #(49972,1)

with open('./txt_CosineSimilarity/LSA_CS_train.txt','w') as file:
    file.write(json.dumps(LSA_CosSim_train))

# # ------------------- test ------------------------
lsa_test = pd.read_csv('X_lsa_test.csv') 
# print(lsa_test) # (50826, 11)
lsa_test = lsa_test.drop(columns=['index'])
# print(lsa_test) # (50826, 10)
lsa_test = lsa_test.to_dict(orient="index")
# print (lsa_test)

# print (cosine_similarity(lsa_test[0],lsa_test[49972]))

LSA_CosSim_test = {}
for i in range(0, 25413):
    LSA_CosSim_test.update({i:cosine_similarity(lsa_test[i],lsa_test[i+25413])})
    if LSA_CosSim_test[i] == None:
        LSA_CosSim_test[i] = 0.0
# print (LSA_CosSim_test) #(25413,1)

with open('./txt_CosineSimilarity/LSA_CS_test.txt','w') as file:
    file.write(json.dumps(LSA_CosSim_test))

