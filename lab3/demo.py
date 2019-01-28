import numpy as np 
#import matplotlib.pyplot as plt 
#import seaborn as sbs 
#import tensorflow as tf 
import sklearn as sk 
import pandas as pd

from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv("bank-additional-full.csv") 
#data = pd.read_csv("bank-additional-full.csv",sep=',') 
#print(data.head(n=5))

df_1 = pd.get_dummies(data)
df_2 = df_1.drop("y_no")
df_dummies = df_2.drop("duration")
print(df_dummies[2:4])

clf = ExtraTreesClassifier()
print("Done")