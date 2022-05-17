# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:03:02 2022

@author: Acer
"""
#%% Imports
import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
    
#%% statics/constants here
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','model.pkl')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
HEART_DATASET = os.path.join(os.getcwd(),'Dataset','heart.csv') 

#%% settings or models loading 
mms_scaler = pickle.load(open(MMS_SAVE_PATH, 'rb'))
model = pickle.load(open(MODEL_SAVE_PATH, 'rb'))

#%% load data
df = pd.read_csv(HEART_DATASET)

#%% streamlit 

# input from user
def user_input(age,sex,cp,trtbps,chol,fbs,restecg,thalach,exng,oldpeak,slp,
               ca,thall):   
    
    if sex=="male":
        sex=1 
    else: 
        sex=0
    
    if exng=="Yes":
        exng=1
    else:
        exng=0
        
    if cp=="typical angina":
        cp=1
    elif cp=="atypical angina":
        cp=2
    elif cp=="non-anginal pain":
        cp=3
    elif cp=="asymptomatic":
        cp=4

    if fbs=="true":
        fbs=1
    else:
        fbs=0
 
    if restecg=="Normal":
        restecg=0
    elif restecg=="having ST-T wave abnormality":
        restecg=1
    elif restecg=="showing probable or definite left ventricular hypertrophy by Estes' criteria":
        restecg=2

    X_test=np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalach,exng,oldpeak,
                      slp,ca,thall]).reshape(1,-1) 
    X_test=mms_scaler.transform(X_test) 
    
    # to do prediction
    y_pred = model.predict(X_test)

    return y_pred

# Title of the form   
st.title('Heart Attack Analysis Prediction')      
# input form 
age=st.slider('Your Age', 0, 150)
sex = st.radio("Gender: ", ('male', 'female'))
cp = st.selectbox('Chest Pain Type',("typical angina","atypical angina","non-anginal pain","asymptomatic")) 
trtbps=int(st.number_input('Resting Blood Sugar in mm Hg'))
chol=st.number_input('Serum Cholestoral in mg/dl')
fbs=st.radio("Fasting Blood Sugar > 120 mg/dl", ['true','false'])
restecg=st.radio('Resting Electrocardiographic Results',("Normal","having ST-T wave abnormality","showing probable or definite left ventricular hypertrophy by Estes' criteria"))
thalach=int(st.number_input('Maximum Heart Rate Achieved'))
exng=st.radio('Exercise Induced Angina',("Yes","No"))
oldpeak=st.number_input('Oldpeak')
slp = st.selectbox('The slope of the peak exercise ST segment',np.sort(df['slp'].unique()))
ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,4))
thall=st.selectbox('Thalium Stress Result',np.sort(df['thall'].unique()))

# submit button
submitted = st.button('Submit')

if submitted:
    y_pred=user_input(age,sex,cp,trtbps,chol,fbs,restecg,thalach,exng,oldpeak,slp,
               ca,thall)
    if y_pred == 0:
        st.write('less chance of heart attack') 
    else:
        st.write('More chance of heart attack')   
        