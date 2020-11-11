import pandas as pd
import numpy as np
import warnings
import random
from random import sample 
from collections import Counter
from datetime import datetime
from copy import deepcopy
warnings.filterwarnings("ignore")

ml1m_dir = '/daintlab/data/movielens/data/ml-1m/ratings.dat'

#load data & userid,itemid Reindex

ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ratings = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

dict2 = dict(zip(ratings["itemId"],ratings["timestamp"]))
items = set(dict2.keys())
num_items = max(dict2.keys()) + 1
print("n_items = {}".format(num_items))
#Binary

ratings['rating'][ratings['rating'] > 0] = 1.0



rating1 = ratings.groupby('userId')['itemId'].apply(list).reset_index()
rating2 = ratings.groupby('userId')['timestamp'].apply(list).reset_index()

ratings = pd.concat([rating1[['userId','itemId']],rating2['timestamp']],axis = 1)



ratings["test_index"] = ratings["timestamp"].apply(lambda x : np.array(x).argmax())

ratings["test_rating"] = ratings.apply(lambda x: x["itemId"][x["test_index"]], axis = 1)
ratings["test"] = ratings["itemId"].apply(lambda x : list(items - set(x)))
ratings["test_negative"] = ratings["test"].apply(lambda x : random.sample(x,99))
ratings["train_negative"] = ratings.apply(lambda x : list(items - set(x["itemId"]) - set(x["test_negative"])), axis = 1)
ratings.apply(lambda x : x["itemId"].remove(x["test_rating"]), axis = 1)
ratings.rename(columns = {"itemId":"train_positive"}, inplace = True)
ratings = ratings[["userId","train_positive","train_negative","test_rating","test_negative"]].reset_index()
print(ratings)






#https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
ratings.to_feather("/daintlab/data/movielens/movie_"+str(num_items)+".ftr")
ratings.to_csv("/daintlab/data/movielens/movie_"+str(num_items)+".csv")