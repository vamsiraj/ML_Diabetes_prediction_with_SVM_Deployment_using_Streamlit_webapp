# -*- coding: utf-8 -*-
"""
Created on Wed May 17 23:17:36 2023

@author: LNT
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/LNT/Desktop/Geeklurn/SVM/trained_model.sav', 'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):
   

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  

def main():
    
    #giving a title 
    st.title('Diabetes Prediction web App')
    
    #Getting the input data from the user
   
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("SkinThickness Value")
    Insulin = st.text_input("Insulin Value")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    Age = st.text_input("Age of the person")
    
    #code for prediction
    diagnosis = ''
    
    #creating a button
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure,
                                         SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()