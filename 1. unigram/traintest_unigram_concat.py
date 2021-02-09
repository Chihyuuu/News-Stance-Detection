import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("merged_traintest.csv")
vectorizer = CountVectorizer() #設定取幾個特徵 max_features=10000

X1 = vectorizer.fit_transform(data.Headline) #fit建構特徵空間; transform使用該空間將數據轉為詞頻矩陣
X1 = pd.DataFrame(X1.toarray(),columns = vectorizer.get_feature_names()) 
# print (X1) # (75385 rows x 4477 columns)
# print (vectorizer.get_feature_names()) #生成的單詞 (list)

X2 = vectorizer.fit_transform(data.articleBody) #fit建構特徵空間; transform使用該空間將文本數據轉為矩陣
X2 = pd.DataFrame(X2.toarray(), columns = vectorizer.get_feature_names()) 
# print (X2) #(75385 rows x 5000 columns) #(75385 rows x 29264 columns)
# # print (vectorizer.get_feature_names()) #生成的單詞 (list)

X = pd.concat([X1,X2],axis=1)
del X1,X2
# X.to_csv('traintest_unigram_unlabeled_10000.csv')
X.to_csv('traintest_unigram_unlabeled.csv')
#5000(75385 rows x 9477 columns)

# X['Stance_4cat'] = data['Stance_4cat']
# print (X)  # all (75385 rows x 33742 columns)
# X.to_csv('traintest_unigram_4cat.csv')
