# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:03:02 2022

@author: Acer
"""
import os
import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
    
#%% statics/constants here
HEART_DATASET = os.path.join(os.getcwd(),'Dataset','heart.csv') 
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','model.pkl')
MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')

#%% loading of settings or models
mms_scaler = pickle.load(open(MMS_SAVE_PATH, 'rb'))
model = pickle.load(open(MODEL_SAVE_PATH, 'rb'))

#%% Streamlit deployment

# to format input of the dataset
def format_input(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exng,oldpeak,slope,ca,thall):   
 
    if sex=="male":
        sex=1 
    else: sex=0
    
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exng=="Yes":
        exng=1
    elif exng=="No":
        exng=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
 
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
 
    if thall=="fixed defect: used to be defect but ok now":
        thall=6
    elif thall=="reversable defect: no proper blood movement when excercising":
        thall=7
    elif thall=="normal":
        thall=2.31

    if restecg=="Normal":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="showing probable or definite left ventricular hypertrophy by Estes' criteria":
        restecg=2


    X_test=np.array([age,sex,cp,trestbps,restecg,chol,fbs,thalach,exng,
                     oldpeak,slope,ca,thall]).reshape(1,-1) 
    X_test=mms_scaler.fit_transform(X_test) 
    
    # to to prediction
    y_pred = model.predict(X_test)

    return y_pred

# Title of the form   
st.title('Heart Attack Analysis Prediction')      
# input from a user
age=st.slider('Insert Your Age', 0, 130)
sex = st.radio("Select Gender: ", ('male', 'female'))
cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
trtbps=int(st.number_input('Resting Blood Sugar in mm Hg'))
chol=st.selectbox('Serum Cholestoral in mg/dl',range(1,1000,1))
fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
restecg=st.selectbox('Resting Electrocardiographic Results',("Normal"," ST-T Wave abnormality","showing probable or definite left ventricular hypertrophy by Estes' criteria"))
thalach=st.selectbox('Maximum Heart Rate Achieved',range(1,300,1))
exng=st.selectbox('Exercise Induced Angina',["Yes","No"])
oldpeak=st.number_input('Oldpeak')
slope = st.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))
ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
thall=st.selectbox('Thalium Stress Result',range(1,8,1))

y_pred=format_input(age,sex,cp,trtbps,restecg,chol,fbs,thalach,exng,oldpeak,slope,ca,thall)


if st.button("Submit"):    
  if y_pred[0] == 0:
    st.error('More chance of heart attack')
    
  else:
    st.success('less chance of heart attack')
    
   