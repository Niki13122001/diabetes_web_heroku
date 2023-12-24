# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:12:54 2023

@author: nikitha
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open('C:/Users/nikitha/Downloads/diabeteswebbapp/trained_model1.sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "The person is non Diabetic."
    else:
        return "The person is Diabetic."
    
def main():
    #giving a title
    st.title('Diabetes Prediction Web App')
   
    #getting the input from user
    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BloodPressure value')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction value')
    Age=st.text_input('Age of the person')
    
    
    #code for prediction 
    diagnosis=''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ =='__main__':
    main()
    
    