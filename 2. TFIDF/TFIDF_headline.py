import csv
import math

with open('processed_headline.csv', newline='', encoding="utf-8") as csvfile:

	dataset = csv.reader(csvfile)
	
	TF_headline = {}
	DF_headline = {}
	doc_num = 0
	for row in dataset:
		#print (row)
		doc_num += 1
		tmpWord = {}
		for word in row:

			if word in TF_headline:
				TF_headline[word] += 1
			else:
				TF_headline[word] = 1

			if word in DF_headline: #別篇出現過
				if word not in tmpWord: #這篇沒出現
					DF_headline[word] += 1
					tmpWord[word] = 1
			else: #別篇沒出現過
				DF_headline[word] = 1

	#print (sorted(TF_headline.items(), key=lambda x:x[1], reverse=True)) #TF倒敘排序
	#print (sorted(DF_headline.items(), key=lambda y:y[1], reverse=True)) #DF倒敘排序

	TFIDF_Headline = {}
	# word_num = 0
	for word in TF_headline:
		# word_num += 1
		TFIDF_Headline[word] = TF_headline[word]*math.log(doc_num/DF_headline[word],10) 
		#print (word, TF_headline[word], DF_headline[word], TFIDF_Headline[word])
	sort_TFIDF_headline = sorted(TFIDF_Headline.items(), key=lambda z:z[1], reverse=True)
	# print ("word_num: ",word_num)
	# print (len(sort_TFIDF_headline)) #4135
	# print (sort_TFIDF_headline)

	count = 0
	headline_worddict = {}
	for key, value in sort_TFIDF_headline:
		count += 1
		headline_worddict[key] = key
		if count >=3000:
			break
	print (headline_worddict)
	print (len(headline_worddict))
