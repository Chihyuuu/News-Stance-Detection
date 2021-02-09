import pandas as pd
import json
from sklearn.feature_extraction import DictVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CSHeadline_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_headline.txt','r',encoding="UTF-8"))
CSBody_TFIDF = json.load(open('./txt_CosineSimilarity/TFIDF_forCS_body.txt','r',encoding="UTF-8"))
LSA_CS_train = json.load(open('./txt_CosineSimilarity/LSA_CS_train.txt','r',encoding="utf-8"))
LSA_CS_test = json.load(open('./txt_CosineSimilarity/LSA_CS_test.txt','r',encoding="utf-8"))
data = pd.read_csv("merged_traintest.csv")

# y = data['Stance_2cat']
y = data['Stance_4cat'] #(75385,1)
# # print ("y",y)

# ------------ feature_TFIDF ------------
dict_vec = DictVectorizer(sparse=True)
X1 = dict_vec.fit_transform(CSHeadline_TFIDF.values())
X1 = pd.DataFrame(X1.toarray(),columns = dict_vec.get_feature_names()) 
# print (X1) # (75385 rows x 3293 columns)

X2 = dict_vec.fit_transform(CSBody_TFIDF.values())
X2 = pd.DataFrame(X2.toarray(),columns = dict_vec.get_feature_names())
# print (X2) # (75385 rows x 4207 columns)

X = pd.concat([X1,X2],axis=1) # (75385 rows x 7500 columns)
# print(X)
del X1,X2

# ------------ feature_LSACS ------------
X3 = pd.DataFrame.from_dict(LSA_CS_train,orient='index',columns=['LSA_CS']) # (49972 x 1)
X4 = pd.DataFrame.from_dict(LSA_CS_test,orient='index',columns=['LSA_CS']) # (25413 x 1)
X_LSACS = pd.concat([X3,X4],axis=0,ignore_index=True)
# print (X_LSACS) #(75385 rows x 1 columns)
del X3,X4

# ------------ feature_TFIDF+LSACS ------------
X = pd.concat([X,X_LSACS],axis=1)
# print (X) # (75385 rows x 7501 columns)

X_train = X.iloc[:49972]
y_train = y.iloc[:49972]

X_test = X.iloc[49972:75385]
y_test = y.iloc[49972:75385]

# ## ------------------MultinomialNB-------------------------
# multiNB = MultinomialNB()
# multiNB.fit(X_train,y_train)
# y_Pred = multiNB.predict(X_test)

# print ("MultinomialNB")
# print ("accuracy:",multiNB.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 agree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
# print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
# print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))

# ## ------------------GaussianNB-------------------------
# GaussianNB = GaussianNB()
# GaussianNB.fit(X_train,y_train)
# y_Pred = GaussianNB.predict(X_test)

# print ("GaussianNB")
# print ("accuracy:",GaussianNB.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 agree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
# print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
# print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))

# ## ------------------DecisionTree--------------------------
# DecisionTree = DecisionTreeClassifier()
# DecisionTree.fit(X_train,y_train)
# y_Pred = DecisionTree.predict(X_test)

# print ("DecisionTree")
# print ("accuracy:",DecisionTree.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 agree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
# print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
# print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))

# ## ------------------RandomForestClassifier-----------------
# RandomForest = RandomForestClassifier()
# RandomForest.fit(X_train,y_train)
# y_Pred = RandomForest.predict(X_test)

# print ("RandomForest")
# print ("accuracy:",RandomForest.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 agree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
# print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
# print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))

## -------------------LinearDiscriminant-------------------
# LinearDiscriminant = LinearDiscriminantAnalysis()
# LinearDiscriminant.fit(X_train,y_train)
# y_Pred = LinearDiscriminant.predict(X_test)

# print ("LinearDiscriminant")
# print ("accuracy:",LinearDiscriminant.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 agree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
# print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
# print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))

# # # -------------------LogisticRegression-------------------
# LogisticRegression = LogisticRegression()
# LogisticRegression.fit(X_train,y_train)
# y_Pred = LogisticRegression.predict(X_test)

# print ("LogisticRegression")
# print ("accuracy:",LogisticRegression.score(X_test,y_test))
# print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
# print ("precision" , "recall", "fscore", "support")
# print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
# print ("1 related: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))

## -------------------- GradientBoosting --------------------
GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
y_Pred = GradientBoosting.predict(X_test)

print ("GradientBoosting")
print ("accuracy:",GradientBoosting.score(X_test,y_test))
print ("confusion_matrix:\n",confusion_matrix(y_test, y_Pred))
print ("precision" , "recall", "fscore", "support")
print ("0 unrelated: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[0]))
print ("1 related: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[1]))
print ("2 disagree: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[2]))
print ("3 discuss: ",precision_recall_fscore_support(y_test, y_Pred, average='micro', labels=[3]))