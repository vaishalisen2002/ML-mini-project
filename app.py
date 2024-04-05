import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('C:/Users/91993/PycharmProjects/diabetes-svm-prediction/model.pkl', 'rb'))

# Define the Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.write("Enter patient details to predict diabetes")

    # Create input fields for user input
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.001)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, step=1)
    bmi = st.number_input("BMI", min_value=0, max_value=70, step=1)
    age = st.number_input("Age", min_value=0, max_value=150, step=1)

    # Create a button to predict the outcome
    if st.button("Predict"):
        # Make prediction using the model
        input_data = [[dpf,pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age]]
        prediction = model.predict(input_data)

        # Display the prediction result
        if prediction[0] == 0:
            st.write("Prediction: No Diabetes")
        else:
            st.write("Prediction: Diabetes")

# Run the app
main()