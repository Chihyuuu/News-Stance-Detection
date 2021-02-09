import pandas as pd
import csv
import nltk

db = pd.read_csv('merged_traintest.csv')

db['lower'] = db['articleBody'].str.lower() #轉小寫
#print (db['lower'])

#對每筆資料做Tokenization
def identify_tokens(row):
    review = row['lower']
    tokens = nltk.word_tokenize(review)   
    # .isalpha 字符串只由字母組成  # .isalnum 字符串由字母和数字组成
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
db['tokenize'] = db.apply(identify_tokens, axis=1)
#print (db['tokenize'])

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
        "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
        "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", 
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
        "about", "against", "between", "into", "through", "during", "before", "after", 
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", "where", 
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
        "very", "s", "t", "can", "will", "just", "don", "should", "now", "the"]

def remove_stops(row):
    my_list = row['tokenize']
    meaningful_words = [w for w in my_list if not w in stopwords]
    return (meaningful_words)
db['meaningful_words'] = db.apply(remove_stops, axis=1)
#print (db['meaningful_words'])

db.to_csv('preprocessed_body.csv', index=False)
#print (list(db)) #列出資料欄位

#二維list寫成CSV
processed = []
for row in db['meaningful_words']:
    processed.append(row)
#print(processed)

with open('processed_body.csv', 'w', newline='',encoding="utf-8") as file:
    wr = csv.writer(file)
    for text in processed:
    	wr.writerow(text)
