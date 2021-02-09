import pandas as pd
import json
import csv

GI = pd.read_excel("myGI_未處理井字.xlsx")
# print (GI) #Dataframe (11789rows x 17columns) 0row:該類別之單詞個數

mydict = {} #裡面放一個單字一個字典，字典的value放類別list

for row in range(1, len(GI)): #每row讀取
	for j in range(0, len(GI.iat[row,0])):#該row第0欄位(單字)
		if(GI.iat[row,0][j] == '#'):
			GI.iat[row,0] = GI.iat[row,0][0:j] #處理包含'#'的單字 僅保留'#'以前的字元
			break

GI_categories=['Positiv','Negativ','Affil','Hostile','Strong','Power','Weak','Submit','Active','Passive','Virtue','Vice','Yes','No','Negate','Intrj']

for row in range(1, len(GI)): #每個row讀取
	for col in GI_categories: #每個(16)類別讀取
		exist = False
		if(not pd.isnull(GI.loc[row,col])): #如果excel欄位有值
			for index in range(0, len(mydict)):
				if(GI.loc[row,'Entry'] in list(mydict.keys())): # GI.loc[row,0]: about, 
					exist = True
					mydict[GI.loc[row,'Entry']].update({GI.loc[row,col]:GI.loc[row,col]})
					break

			if(not exist):
				word_dict = {}
				category = set()
				category.add(GI.loc[row,col])
				category = {word:word for word in category}
				word_dict.update({GI.loc[row,'Entry']:category})
				mydict.update(word_dict)
# print (mydict)
# with open('feature_GI.txt','w') as file:
# 	file.write(json.dumps(mydict))

#--------------處理為以類別為主的雙層字典---------------
# GI = pd.read_csv("feature_GI.csv")
# GI_dict = {}
# GI_categories=['Positiv','Negativ','Affil','Hostile','Strong','Power','Weak','Submit','Active','Passive','Virtue','Vice','Yes','No','Negate','Intrj']

# for i in GI_categories:
# 	GI_dict.update({i:{}})
	
# 	for row in range(0,5402):
# 		if GI.loc[row,i] == i:
# 			# print(GI.loc[row,'Entry'])
# 			GI_dict[i].update({GI.loc[row,'Entry']:0})

# with open('GI_dict.txt','w') as file:
# 	file.write(json.dumps(GI_dict))


# #---------------讀取處理過的body文本 計算每個類別的字數--------------
# GI_dict = json.load(open('GI_dict.txt','r',encoding="utf-8"))
# # # print (len(GI_dict)) #16類別
# # # print (GI_dict)

# with open('processed_body.csv', newline='', encoding="utf-8") as csvfile:

# 	body = csv.reader(csvfile)

# 	body_GI = {}
# 	index = 0

# 	for row in body: #逐筆讀取body
# 		body_GI.update({index:{}})

# 		for category in GI_dict.keys(): #16類別 (positiv,negativ,...)
# 			body_GI[index].update({category:0})
# 		# print (body_GI[index])
# 		# break
# 			count=0
# 			for word in row: #一筆body的每個字 
# 				if word in GI_dict[category].keys(): #判斷字是否出現在該類別的字中
# 					count += 1
# 			body_GI[index].update({category:count})
# 		index += 1 

# 	# print (body_GI)

# with open('feature_body_GI.txt','w') as file:
# 	file.write(json.dumps(body_GI))