import csv
import math
import json
import pandas as pd
from TFIDF_headline import DF_headline
from TFIDF_headline import doc_num

hb_worddict = json.load(open('hb_worddict.txt','r',encoding="utf-8"))
# print (len(hb_worddict)) #4282

with open('processed_headline.csv', newline='', encoding="utf-8") as csvfile:

	Headline = csv.reader(csvfile)
	
	CSHeadline_TFIDF = {}

	index = 0
	for row in Headline:
		CSHeadline_TFIDF.update({index:{}})

		# 1.計算TF
		for word in row: # 一筆Headline中的每個字
					
			if word in hb_worddict:
				# print(CSHeadline_TFIDF[index])
				# print(CSHeadline_TFIDF[index].get(word))

				if CSHeadline_TFIDF[index].get(word): # 拿word的value, 判斷word的value有值
					CSHeadline_TFIDF[index][word] += 1
					# print(CSHeadline_TFIDF[index][word])
				else :
					CSHeadline_TFIDF[index].update({word:1})
		# print(CSHeadline_TFIDF[index])

		# 2.計算TFIDF
		for keyword in  CSHeadline_TFIDF[index]:
			CSHeadline_TFIDF[index][keyword] *= math.log(doc_num/DF_headline[keyword],10)
		# print (CSHeadline_TFIDF[index].keys())
	
		index += 1
	# print(CSHeadline_TFIDF) #75385筆
# with open('./txt_CosineSimilarity/TFIDF_forCS_headline.txt','w') as file:
# 	file.write(json.dumps(CSHeadline_TFIDF))
