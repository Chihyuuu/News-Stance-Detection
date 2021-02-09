import json
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer

CSHeadline_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_headline.txt','r',encoding="UTF-8"))
CSBody_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_body.txt','r',encoding="UTF-8"))

dict_vec = DictVectorizer(sparse=True)
Headline_TFIDF = dict_vec.fit_transform(CSHeadline_TFIDF.values())
Headline_TFIDF = pd.DataFrame(Headline_TFIDF.toarray(),columns = dict_vec.get_feature_names()) 
# print (Headline_TFIDF) # (75385 rows x 3293 columns)
X1 = Headline_TFIDF.iloc[:49972] #僅取train data 丟進lda model做訓練
# print (X1) # (49972 rows x 3293 columns)

Body_TFIDF = dict_vec.fit_transform(CSBody_TFIDF.values())
Body_TFIDF = pd.DataFrame(Body_TFIDF.toarray(),columns = dict_vec.get_feature_names())
# print (Body_TFIDF) # (75385 rows x 4207 columns)
X2 = Body_TFIDF.iloc[:49972] #僅取train data 丟進lda model做訓練
# print (X2) # (49972 rows x 4207 columns)

X_train = pd.concat([X1,X2],axis=0,join='outer',ignore_index=True).fillna(0)
# print(X_train) # (99944 rows x 4282 columns)
# X_train.to_csv('train_TFIDF_forlda.csv')

lda = LatentDirichletAllocation(n_components=10)
# # lda = LatentDirichletAllocation(n_components=20)
lda.fit(X_train)
X_lda_train = lda.transform(X_train)
# print (X_lda_train) #(99944 rows x 10 columns)
del X_train
pd.DataFrame(X_lda_train).to_csv('X_lda_train.csv')

X3 = Headline_TFIDF.iloc[49972:]
X4 = Body_TFIDF.iloc[49972:]
# print (X3) # (25413 rows x 3293 columns)
# print (X4) # (25413 rows x 4207 columns)
X_test = pd.concat([X3,X4],axis=0,join='outer',ignore_index=True).fillna(0)
# print (X_test) # (50826 rows x 4282 columns)
del X3,X4

X_lda_test = lda.transform(X_test)
del X_test
# print (X_lda_test) #(50826, 10)
pd.DataFrame(X_lda_test).to_csv('X_lda_test.csv')
