import csv
import math
import json
import pandas as pd
from TFIDF_body import DF_body
from TFIDF_body import doc_num

hb_worddict = json.load(open('hb_worddict.txt','r',encoding="utf-8"))

with open('processed_body.csv', newline='', encoding="utf-8") as csvfile:

	Body = csv.reader(csvfile)
	
	CSBody_TFIDF = {}

	index = 0
	for row in Body:
		CSBody_TFIDF.update({index:{}})

		# 1.計算TF
		for word in row: # 一筆Headline中的每個字
					
			if word in hb_worddict:
				# print(CSBody_TFIDF[index])
				# print(CSBody_TFIDF[index].get(word))

				if CSBody_TFIDF[index].get(word): # 拿word的value, 判斷word的value有值
					CSBody_TFIDF[index][word] += 1
					# print(CSBody_TFIDF[index][word])
				else :
					CSBody_TFIDF[index].update({word:1})
		# print(CSBody_TFIDF[index])

		# 2.計算TFIDF
		for keyword in  CSBody_TFIDF[index]:
			CSBody_TFIDF[index][keyword] *= math.log(doc_num/DF_body[keyword],10)

		# print (CSBody_TFIDF)
		# break		
		index += 1
	# print(CSBody_TFIDF) #75385筆
# with open('./txt_CosineSimilarity/TFIDF_forCS_body.txt','w') as file:
# 	file.write(json.dumps(CSBody_TFIDF))
