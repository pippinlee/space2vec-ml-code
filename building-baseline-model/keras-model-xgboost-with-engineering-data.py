
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from scipy.ndimage import imread
from keras.utils import np_utils
from space_utils import metrics
from time import process_time
from astropy.io import fits
from os.path import isfile
from keras import backend

import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np

import pickle
import random
import json
import sys
import os

pd.set_option("max_columns", 999)

np.random.seed(1)

get_ipython().magic(u'matplotlib inline')


# In[2]:


# Load engineered data from pickle
data = pickle.load(open('/home/ubuntu/transients-data-processing/data/engineered-data.pkl', 'rb'))


# In[3]:


targets = [
    "OBJECT_TYPE",
]

ids = [
    "ID",
]

continuous = [
    "AMP",
    "A_IMAGE",
    "A_REF",
    "B_IMAGE",
    "B_REF",
    "COLMEDS",
    "DIFFSUMRN",
    "ELLIPTICITY",
    "FLUX_RATIO",
    "GAUSS",
    "GFLUX",
    "L1",
    "LACOSMIC",
    "MAG",
    "MAGDIFF",
    "MAG_FROM_LIMIT",
    "MAG_REF",
    "MAG_REF_ERR",
    "MASKFRAC",
    "MIN_DISTANCE_TO_EDGE_IN_NEW",
    "NN_DIST_RENORM",
    "SCALE",
    "SNR",
    "SPREADERR_MODEL",
    "SPREAD_MODEL",
]

categorical = [
    "BAND",
    "CCDID",
    "FLAGS",
]

ordinal = [
    "N2SIG3",
    "N2SIG3SHIFT",
    "N2SIG5",
    "N2SIG5SHIFT",
    "N3SIG3",
    "N3SIG3SHIFT",
    "N3SIG5",
    "N3SIG5SHIFT",
    "NUMNEGRN",
]

booleans = [
    "MAGLIM",
]

# continuous = [c for c in columns if c not in (special + categorical + ordinal + booleans)]


# In[4]:


# One-hot encode categorical
data = pd.get_dummies(
    data, 
    prefix = categorical, 
    prefix_sep = '_',
    dummy_na = True, 
    columns = categorical, 
    sparse = False, 
    drop_first = False
)


# In[5]:


target = data[targets]
inputs = data.drop(columns=ids+targets)


# In[6]:


# Shuffle and split the data
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42, stratify=target.as_matrix())


# In[7]:


train_x, train_y, valid_x, valid_y = X_train.as_matrix(), y_train.as_matrix(), X_test.as_matrix(), y_test.as_matrix()

# save dmatrices
xgtrain = xgb.DMatrix(train_x, label=train_y, feature_names=X_train.columns)
xgvalid = xgb.DMatrix(valid_x, label=valid_y, feature_names=X_test.columns)


# In[31]:


param = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'silent': 1,
    'objective': 'binary:logistic',
    'scale_pos_weight': 0.5,
    'n_estimators': 40,
    "gamma": 0,
    "min_child_weight": 1,
    "max_delta_step": 0, 
    "subsample": 0.9, 
    "colsample_bytree": 0.8, 
    "colsample_bylevel": 0.9, 
    "reg_alpha": 0, 
    "reg_lambda": 1, 
    "scale_pos_weight": 1, 
    "base_score": 0.5,  
    "seed": 23,  
}

param['nthread'] = 4
blah_metric = ['error', 'auc']

evallist = [(xgtrain, 'train'), (xgvalid, 'valid')]

bst = xgb.XGBClassifier(**param)

bst. train_x, train_y), (valid_x, valid_y)], 
    eval_metric=blah_metric, 
    verbose=True
)

# clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=’logloss’, verbose=True

# bst = xgb.train(param, xgtrain, evals=evallist, num_boost_round=param['num_round'])

# bst.save_model('xgb_' + str(2018) + '_v1.model')


# In[54]:


# ytrue = xgvalid.get_label()
ytrue = valid_y


# bst = xgb.Booster({'nthread': 4}) #init model
# bst.load_model("xgb_" + str(2018) + "_v0.model") # load data

ypred = bst.predict_proba(valid_x)[:, 1:]


# In[36]:


a = bst.predict_proba(valid_x)


# In[49]:


print(ytrue.shape)
print(ypred.shape)


# In[55]:


metrics(ypred, ytrue, threshold=0.5)


# In[56]:


ypred

