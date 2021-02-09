import pandas as pd 
import csv

list1 = pd.read_csv('merge_traindata.csv', sep=",") 
list2 = pd.read_csv('merge_testdata.csv', sep=",") 

merge = pd.concat([list1, list2],axis=0,ignore_index=True)
# print (merge)
merge.to_csv('merged_traintest.csv')
