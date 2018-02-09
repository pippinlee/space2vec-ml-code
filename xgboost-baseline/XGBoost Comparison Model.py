
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split

import xgboost as xgb
import pandas as pd
import numpy as np

import pickle
import random

pd.set_option("max_columns", 999)

np.random.seed(1)


# ## Let's get started!
# 
# First we have to load in the data, this is the feature engineered data right from the paper. We have actually taken the extra step of formatting it really nicely for Python.
# 
# Make sure to change the path to where you downloaded the data!

# In[2]:


path_to_data = "/Users/clifford-laptop/Documents/space2vec/data/engineered-data.pkl"

data = pickle.load(open(path_to_data, 'rb'))


# ## Next the column types
# 
# Not all of this is necessary but we wanted to make sure that we explicitly state what each column type is. That way we can be sure that we don't include columns that shouldn't be in the training data.

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


# ## One hot encode any categorical columns
# 
# Here we do something called one hot encoding (https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f).
# 
# This is to turn any categorical columns into something that a machine learning model can understand. Let's say we have a column, maybe we call it BAND, and this column might have 4 different possible values:
# 
# g, i, r, or z
# 
# Well we can't really shove these into our network so we hit it with the "one hot"! The BAND column becomes 5 different columns:
# 
# BAND_g, BAND_i, BAND_r, BAND_z, and BAND_nan
# 
# Now, instead of a letter value; we have a binary representation with a 1 in it's corresponding column and a zero in the rest.
# 
# The function is a bit interesting but it does exactly what we need!

# In[4]:


data = pd.get_dummies(
    data, 
    prefix = categorical, 
    prefix_sep = '_',
    dummy_na = True, 
    columns = categorical, 
    sparse = False, 
    drop_first = False
)


# ## Split the inputs from the targets
# 
# This is super important!
# 
# We have to make sure we physically seperate the targets (aka labels) from our model input. This is to give us a piece of mind as we train.
# 
# Obviously, the model should never train on our targets... That's like giving a student the exam answer sheet to study before the exam!

# In[5]:


target = data[targets]
inputs = data.drop(columns = ids + targets)


# ## Shuffle and split the data
# 
# Now we split the data again, this time into a training set and a validation set.
# 
# This is comparable to having a bunch of practice questions before a test (the training set) and quiz questions (the validation set).
# 
# **It's important to note that the model should never learn on the validation set!**
# 
# We also shuffle the data to make sure we remove any possible patterns that could be happening within the data (not very likely to happen in this dataset but it doesn't hurt).
# 
# Another **really** important point here is "stratification". That sounds fancy but it basically means that when we split the data, the distribution of the populations should be the same in the training and validation set as it was originally... That didn't help did it?
# 
# Let's say that in the total dataset we have 50.5% of the population as supernova and the other 49.5% of the population being not a supernova. When we split the data into two subset, in a stratified way, both subsets should keep a very similar ratio of supernova to not-supernova (50.5% to 49.5%).
# 
# This is getting way too long... Lastly I'll point out the **test_size = 0.2**. This simply means that 20% of the data is put into a validation set (leaving the other 80% as training data).

# In[9]:


x_train, x_valid, y_train, y_valid = train_test_split(
    inputs, 
    target, 
    test_size = 0.2, 
    random_state = 42,
    stratify = target.as_matrix()
)


# ## Parameters!
# 
# Alright, we won't get too into the specifics here but you can definitely check out the documentation (http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier).
# 
# We just toyed around with the parameters to see what seemed to work the best.
# 
# Once we get to the Convolutional Neural Network (CNN), the model we will more than likely use in the end, we will automate this parameter search.
# 
# **The joys of this whole notebook thing is that you can run all of this! Try changing them and see what happens!**

# In[26]:


params = {
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
    "nthread": 4
}


# ## *Rocky training montage*
# 
# Now for the part where Rocky runs through the streets training for the big fight!
# 
# Ahaha, oh the joys of modern programming! All we need to do is define the XGBClassifier and `.fit()`!
# 
# As long as we pass in the data and the metrics that we want to define then we are good to go.

# In[27]:


bst = xgb.XGBClassifier(**params)

bst.fit(
    x_train, 
    y_train, 
    eval_set = [(x_train, y_train), (x_valid, y_valid)], 
    eval_metric = ['auc'], 
    verbose = True
)


# ## Define the rules of the ring
# 
# The rules of the big finale were described within the paper, these are the Missed Detection Rate (MDR) and the False Positive Rate (FPR). We won't dive in here as they are mentioned in depth in our blog post, but the following is the coded version of the metrics.

# In[31]:


def metrics(outputs, labels, threshold=0.5):
    predictions = outputs >= (1 - threshold)
    true_positive_indices = (predictions == 0) * (labels == 0)
    false_positive_indices = (predictions == 0) * (labels == 1)
    true_negative_indices = (predictions == 1) * (labels == 1)
    false_negative_indices = (predictions == 1) * (labels == 0)

    true_positive_count = true_positive_indices.sum()
    false_positive_count = false_positive_indices.sum()
    true_negative_count = true_negative_indices.sum()
    false_negative_count = false_negative_indices.sum()
   
    return {
        # Missed detection rate
        'MDR': false_negative_count / (true_positive_count + false_negative_count),
        # True positive rate
        'FPR': false_positive_count / (true_negative_count + false_positive_count)
    }


# ## Hiring the referee
# 
# Great, now we have the rules for the big fight. But we also need someone (or something... or just a function) to take action on the rules.
# 
# This is just a function that will run MDR and FPR on all 3 thresholds (0.4, 0.5, 0.6) and a few extras explained below:
# 
# **FALSE_POSITIVE_RATE:** Is the sum of the FPR from all three thresholds, this helps us see how the models compare on a large scale.
# 
# **MISSED_DETECTION_RATE:** Is the sum of the MDR from all three thresholds, this helps us see how the models compare on a large scale.
# 
# **PIPPIN_METRIC:** Named after team member Pippin Lee, this is just **FALSE_POSITIVE_RATE** and **MISSED_DETECTION_RATE** summed to give us an even large scale of how the models compare.
# 
# **ACCURACY:** Simply the percentage of guesses that we got right.

# In[30]:


def get_metrics(outputs, labels, with_acc=True):
    
    all_metrics = {}
    
    # FPR and MDR 0.4
    temp = metrics(outputs, labels, threshold=0.4)
    all_metrics["FALSE_POSITIVE_RATE_4"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_4"] = temp["MDR"]
    
    # FPR and MDR 0.5
    temp = metrics(outputs, labels, threshold=0.5)
    all_metrics["FALSE_POSITIVE_RATE_5"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_5"] = temp["MDR"]
    
    # FPR and MDR 0.6
    temp = metrics(outputs, labels, threshold=0.6)
    all_metrics["FALSE_POSITIVE_RATE_6"] = temp["FPR"]
    all_metrics["MISSED_DETECTION_RATE_6"] = temp["MDR"]
    
    # Summed FPR and MDR
    all_metrics["FALSE_POSITIVE_RATE"] = all_metrics["FALSE_POSITIVE_RATE_4"] + all_metrics["FALSE_POSITIVE_RATE_5"] + all_metrics["FALSE_POSITIVE_RATE_6"] 
    all_metrics["MISSED_DETECTION_RATE"] = all_metrics["MISSED_DETECTION_RATE_4"] + all_metrics["MISSED_DETECTION_RATE_5"] + all_metrics["MISSED_DETECTION_RATE_6"]
    
    # The true sum
    all_metrics["PIPPIN_METRIC"] = all_metrics["FALSE_POSITIVE_RATE"] + all_metrics["MISSED_DETECTION_RATE"]
    
    # Accuracy
    if with_acc:
        predictions = np.around(outputs).astype(int)
        all_metrics["ACCURACY"] = (predictions == labels).sum() / len(labels)
    
    return all_metrics


# ## The big fight!
# 
# Our model has trained up in the modern day version of a classic cinematic training montage!
# 
# We can finally give it the final challange... this challenge just happens to be feeding it more data rather than fighting his own inner demons in the manifestation of a boxer.

# In[36]:


y_predictions = bst.predict_proba(x_valid)[:, 1:]


# ## To the judges!
# 
# Our model has fought well and forced the match to decision. Only the judges can give us the final results!
# 
# You can see that we use the metric functions defined above, passing in what the model guessed and what the actual results **should be**. We then do the math and see how our fighter did.
# 
# We won't go in depth into the comparison here since we go into it in-depth in the article. 
# 
# (Teaser: it lost but actually did fairly well for how simple it is!)

# In[ ]:


all_metrics = get_metrics(y_predictions, y_valid)

print("FPR (0.4): " + str(all_metrics["FALSE_POSITIVE_RATE_4"][0]))
print("FPR (0.5): " + str(all_metrics["FALSE_POSITIVE_RATE_5"][0]))
print("FPR (0.6): " + str(all_metrics["FALSE_POSITIVE_RATE_6"][0]))
print("")
print("MDR (0.4): " + str(all_metrics["MISSED_DETECTION_RATE_4"][0]))
print("MDR (0.5): " + str(all_metrics["MISSED_DETECTION_RATE_5"][0]))
print("MDR (0.6): " + str(all_metrics["MISSED_DETECTION_RATE_6"][0]))
print("")
print("SUMMED FPR: " + str(all_metrics["FALSE_POSITIVE_RATE"][0]))
print("SUMMED MDR: " + str(all_metrics["MISSED_DETECTION_RATE"][0]))
print("TOTAL SUM: " + str(all_metrics["PIPPIN_METRIC"][0]))
print("")
print("ACCURACY: " + str(all_metrics["ACCURACY"][0]))

