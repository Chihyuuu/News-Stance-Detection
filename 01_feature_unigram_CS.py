import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support

X = pd.read_csv("traintest_unigram_unlabeled.csv")
CosineSimilarity = json.load(open('./txt_CosineSimilarity/TFIDF_CosineSimilarity.txt','r',encoding="utf-8"))
data = pd.read_csv("merged_traintest.csv")

X3 = pd.DataFrame.from_dict(CosineSimilarity,orient='index',columns=['CosineSimilarity']) #(75385筆, 1欄)
X['CosineSimilarity'] = list(X3.CosineSimilarity) # (75385筆, 7210欄)
del X3
# print ("X",X.shape)

y = data['Stance_4cat']

X_train = X.iloc[:49972]
y_train = y.iloc[:49972]

X_test = X.iloc[49972:75385]
y_test = y.iloc[49972:75385]
del X,y

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