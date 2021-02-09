import pandas as pd
import json
import csv

# # -1-3------處理類別與類別編號 #{1: {'funct'}, 2: {'pronoun'},...,464: {'filler'}}
# LIWC = pd.read_excel('LIWC_only_categories.xlsx')
# # print (LIWC) # [64 rows x 2 columns]

# LIWC_cat = {}
# for row in range(0,len(LIWC)):
# 	# print (LIWC.loc[row,'index'],LIWC.loc[row,'categories'])
# 	LIWC_cat.update({str(LIWC.loc[row,'index']):{}})
# # print (LIWC_cat)

# # -2------------------------- 處理掉*字號 --------------------------
# LIWC_only_words = pd.read_excel("LIWC_only_words.xlsx")
# # print (LIWC_only_words) #[4487 rows x 9 columns]

# for row in range(1, len(LIWC_only_words)): #每row讀取
# 	for j in range(0, len(LIWC_only_words.iat[row,0])):#該row第0欄位(單字)
# 		if(LIWC_only_words.iat[row,0][j] == '*'):
# 			LIWC_only_words.iat[row,0] = LIWC_only_words.iat[row,0][0:j] #處理包含'*'的單字 僅保留'*'以前的字元
# 			break
# print (LIWC_only_words)
# LIWC_only_words.to_excel("LIWC_only_words2.xlsx")

# # -3--------建造雙層字典 {類別編號:{'word','word'...},...}
# LIWC_only_words = pd.read_csv("LIWC_only_words.csv")
# for row in range(0, len(LIWC_only_words)): #每row讀取
# 	for col in range(1,9): #每個column讀取
# 		if(not pd.isnull(LIWC_only_words.iat[row,col])): #判斷該欄位有值
# 			# print(row)
# 			# print(int(LIWC_only_words.iat[row,col]))
# 			LIWC_cat[str(int(LIWC_only_words.iat[row,col]))].update({LIWC_only_words.iat[row,0]:0})
		
# print(LIWC_cat)
# with open('LIWC_dict.txt','w') as file:
# 	file.write(json.dumps(LIWC_cat))

#-4--------------讀取處理過的body文本 計算每個類別的字數--------------
LIWC_dict = json.load(open('LIWC_dict.txt','r',encoding="utf-8"))
# print (len(LIWC_dict)) #64類別
# print (LIWC_dict)

with open('processed_body.csv', newline='', encoding="utf-8") as csvfile:

	body = csv.reader(csvfile)

	body_LIWC = {}
	index = 0

	for row in body: #逐筆讀取body
		body_LIWC.update({index:{}})

		for category in LIWC_dict.keys(): #16類別 (positiv,negativ,...)
			body_LIWC[index].update({category:0})
		# print (body_LIWC[index])
		# break
			count=0
			for word in row: #一筆body的每個字 
				if word in LIWC_dict[category].keys(): #判斷字是否出現在該類別的字中
					count += 1
			body_LIWC[index].update({category:count})
		index += 1 

	# print (body_LIWC)

with open('feature_body_LIWC.txt','w') as file:
	file.write(json.dumps(body_LIWC))