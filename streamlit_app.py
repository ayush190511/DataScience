import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("artifacts/model.pkl", "rb"))

st.title("🎓 Student Math Score Prediction")

gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])
reading = st.number_input("Reading Score", 0, 100)
writing = st.number_input("Writing Score", 0, 100)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "race_ethnicity": race,
        "parental_level_of_education": parent_edu,
        "lunch": lunch,
        "test_preparation_course": prep,
        "reading_score": reading,
        "writing_score": writing
    }])

    prediction = model.predict(input_df)
    st.success(f"Predicted Math Score: {prediction[0]:.2f}")