from TFIDF_headline import headline_worddict
from TFIDF_body import body_worddict
import json

# UNION
hb_wordset = (set(headline_worddict).union(set(body_worddict)))
hb_worddict = {word:0 for word in hb_wordset}
# print ("hb_worddict", hb_worddict)
# print (len(hb_worddict))

with open('hb_worddict.txt','w') as file:
	file.write(json.dumps(hb_worddict))