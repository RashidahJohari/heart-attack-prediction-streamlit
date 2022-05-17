# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:27:00 2022

@author: Acer
"""
#%% Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import os
import pickle

#%% statics/constants here
HEART_DATASET = os.path.join(os.getcwd(),'Dataset','heart.csv') 
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','model.pkl')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')

#%% EDA
# Step 1) Data loading
# To load data
df = pd.read_csv(HEART_DATASET)

#%%
# Step 2) Data interpretation/inspection
# to check first 5 rows
print(df.head())

# to check number of entries
print (df.shape)

# to check outlier (mean & median comparison)
print(df.describe())
print(df.groupby('output').hist(figsize=(9,9)))

# to check null
print(df.info())

# to check missing data
#print(df.isna().sum())
print(df.isnull().sum())

# to check duplicate values
print(df[df.duplicated()])

# to sort values
print(np.sort(df))

#%%
# Step 3) Data cleaning
# to check duplicate value
#print(df.duplicated().sum())
#print(df.drop_duplicates())
print(df.drop_duplicates(inplace=True))
# to check back if still containing duplicate value
print(df.duplicated().sum())
# inplace=True to remove duplicates from the original DataFrame.

# a) to convert all string data into numerical data
# b) to remove NaN and to impute using some approaches
# b(i) drop NaN data
# b(ii) imputed NaN with mean/median/interpolation : median(recommended)

#%% Step 4) Features selection (correlation/lasso)
#%% Step 5) Data preprocessing (keluar data yg nak guna shj & concat)
# convert all non-numerical/string to number using 
# a) np.to_numeric
# b) label encoder
# c) one hot encoder
# Scaling data using
# a) standardscaler()
# b) MinMaxScaler

X = df.iloc[:,:-1]
y = df['output']

# MinMaxScaler
mms_scaler = MinMaxScaler()
X_scaled = mms_scaler.fit_transform(X)
# save the scaler
pickle.dump(mms_scaler, open(MMS_SAVE_PATH, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=123,stratify=y)

#%% Machine learning model
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_true = y_test

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))
print(round(accuracy_score(y_true, y_pred)*100,2))

#%% Model deployment
model_pkl = pickle.dump(model, open(MODEL_SAVE_PATH, 'wb'))

# deeplearning: model.save(MODEL_SAVE_PATH)
# machinelearning: pickle.dump(model,open(MODEL_SAVE_PATH))

