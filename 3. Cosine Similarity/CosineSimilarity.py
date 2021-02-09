from math import *
import json 
# import pandas as pd
# from sklearn.model_selection import cross_val_score,KFold

CSHeadline_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_headline.txt','r',encoding="utf-8"))
CSBody_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_body.txt','r',encoding="utf-8"))

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

# print (CSHeadline_TFIDF['0'].get())
# CSHeadline_TFIDF = {int(k):v for k,v in CSHeadline_TFIDF.items()}
# CSBody_TFIDF = {int(k):v for k,v in CSBody_TFIDF.items()}

CosineSimilarity = {}
for i in range(0, len(CSHeadline_TFIDF)):
    CosineSimilarity.update({i:cosine_similarity(CSHeadline_TFIDF[str(i)],CSBody_TFIDF[str(i)])})
    if CosineSimilarity[i] == None:
        CosineSimilarity[i] = 0.0
print (CosineSimilarity)

# with open('./txt_CosineSimilarity/TFIDF_CosineSimilarity.txt','w') as file:
#     file.write(json.dumps(CosineSimilarity))
