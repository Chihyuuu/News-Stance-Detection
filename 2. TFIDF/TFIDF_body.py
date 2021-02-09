import csv
import math

with open('processed_body.csv', newline='', encoding="utf-8") as csvfile:

	dataset = csv.reader(csvfile)
	
	TF_body = {}
	DF_body = {}
	doc_num = 0
	for row in dataset:
		#print (row)
		doc_num += 1
		tmpWord = {}
		for word in row:

			if word in TF_body:
				TF_body[word] += 1
			else:
				TF_body[word] = 1

			if word in DF_body: #別篇出現過
				if word not in tmpWord: #這篇沒出現
					DF_body[word] += 1
					tmpWord[word] = 1
			else: #別篇沒出現過
				DF_body[word] = 1
	
	#print (sorted(TF_body.items(), key=lambda x:x[1], reverse=True)) #TF倒敘排序
	#print (sorted(DF_body.items(), key=lambda y:y[1], reverse=True)) #DF倒敘排序

	TFIDF_body = {}
	word_num = 0
	for word in TF_body:
		word_num += 1
		TFIDF_body[word] = TF_body[word]*math.log(doc_num/DF_body[word],10) 
		#print (word, TF_body[word], DF_body[word], TFIDF_body[word])
	sort_TFIDF_body = sorted(TFIDF_body.items(), key=lambda z:z[1], reverse=True)
	#print ("word_num: ",word_num)
	# print (len(sort_TFIDF_body)) #26999

	count = 0
	body_worddict = {}
	for key, value in sort_TFIDF_body:
		count += 1
		body_worddict[key] = key
		if count >=3000:
			break
	#print (body_worddict)
	#print (len(body_worddict))
