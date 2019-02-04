import pandas as pd
import numpy as np
from IPython.display import Image
np.set_printoptions(precision = 3)

user_ratings_df = pd.read_csv("user_ratings.csv")
user_features_df = pd.read_csv("user_features.csv")
item_features_df = pd.read_csv("item_features.csv")


user_features_df["key"] = 0
user_features_df["user_id"] = range(0,user_features_df.shape[0])
item_features_df["key"] = 0
item_features_df["item_id"] = range(0,item_features_df.shape[0])

merged_df = pd.merge(user_features_df, item_features_df,left_index=True,on="key")
merged_df[["item_id", "user_id"]]




merged_df["rating"] = map(lambda ids: user_ratings_df.values[ids[1]][ids[2]], 
                          merged_df[["user_id", "item_id"]].itertuples())

train = merged_df.dropna()

test = merged_df[merged_df.isnull().any(axis=1)]

print (test.to_latex())

