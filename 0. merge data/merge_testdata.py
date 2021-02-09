import pandas as pd 
import csv

# list1 = pd.read_csv('competition_test_stances.csv', sep=",") 
# list2 = pd.read_csv('competition_test_bodies.csv', sep=",") 

# merge = pd.merge(list1, list2,on='Body ID')
# merge.to_csv('merge_testdata.csv')

data = pd.read_csv("merge_testdata.csv")

# Stance_2cat = []
# for row in data['Stance']:
# 	if row == "unrelated":
# 		Stance_2cat.append(0)
# 	else:
# 		Stance_2cat.append(1)
# data['Stance_2cat'] = Stance_2cat

# Stance_4cat = []
# for row in data['Stance']:
# 	if row == "unrelated":
# 		Stance_4cat.append(0)
# 	elif row == "agree":
# 		Stance_4cat.append(1)
# 	elif row == "disagree":
# 		Stance_4cat.append(2)
# 	else: #discuss
# 		Stance_4cat.append(3)
# data['Stance_4cat'] = Stance_4cat

# ##計算2個類別筆數
# count_unrelated=0
# count_related=0
# for row in data['Stance']:
# 	if row == "unrelated":
# 		count_unrelated+=1
# 	else:
# 		count_related+=1
# print ("unrelated",count_unrelated,"related",count_related)

## 計算4個類別筆數
count_unrelated=0
count_agree=0
count_disagree=0
count_discuss=0
for row in data['Stance']:
	if row == "unrelated":
		count_unrelated+=1
	elif row == "agree":
		count_agree+=1
	elif row == "disagree":
		count_disagree+=1
	else: #discuss
		count_discuss+=1
print ("unrelated",count_unrelated,"agree",count_agree,"disagree",count_disagree,"discuss",count_discuss)

# data.to_csv("merge_testdata.csv",index=False)