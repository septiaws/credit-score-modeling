# Import library
import streamlit as st
import requests
import pandas as pd

# Create the title
st.title("Credit Score Prediction")
# st.subheader("Input the applicant's data and click the "Predict" button")

# Create the form fpr input
with st.form(key = "applicant_data_form"):

    # Input applicant's name
    app_name = st.text_input('Applicants name', '')

    # Input person_income
    person_income = st.number_input(
        label = "1.\tTotal annual income :",
        min_value = 10500,
        max_value = 1000000,
        help = "Value range from 10500 to 1000000"
    )

    # Input person_age
    person_age = st.number_input(
        label = "1.\tAge:",
        min_value = 20,
        max_value = 55,
        help = "Value range from 20 to 55"
    )

    # Input person_experience
    person_experience = st.number_input(
        label = "1.\tProfessional experience in years:",
        min_value = 0,
        max_value = 20,
        help = "Value range from 0 to 20"
    )

    # Input current_job_years
    current_job_years = st.number_input(
        label = "1.\tcurrent job in years:",
        min_value = 0,
        max_value = 14,
        help = "Value range from 0 to 14"
    )

    # Input current_house_years
    current_house_years = st.number_input(
        label = "1.\tcurrent house in years:",
        min_value = 10,
        max_value = 14,
        help = "Value range from 10 to 14"
    )

    # Input married
    married = st.selectbox(
        label = "9. \tmarried:",
        options = ("married", "single")
    )

    # Input house_ownership
    house_ownership = st.selectbox(
        label = "9. \thouse ownership:",
        options = ("norent_noown", "owned", "rented")
    )

    # Input car_ownership
    car_ownership = st.selectbox(
        label = "9. \tcar ownership:",
        options = ("no", "yes")
    )

    # Input profession
    profession = st.radio(
        label = "8. \tProfession:",
        options = ["Analyst", "Architect", "Artist", "Biomedical Engineer", "Chef", "Civil Servant",
                   "Consultant", "Designer", "Economist", "Engineer", "Fashion Designer", "Graphic_Designer",
                   "Lawyer", "Mechanical Engineer", "Scientist", "Statistician"],
        index = 0,
        horizontal = True
    )

    # Input city
    city = st.radio(
        label = "8. \tCity:",
        options = ["Agra", "Ahmednagar", "Akola", "Amaravati", "Belgaum", "Bhiwani",
                   "Chapra", "Dehradun", "Dharmavaram", "Gangtok", "Hapur", "Jamalpur", "Lawyer", "Vellore"],
        index = 0,
        horizontal = True
    )

    # Input state
    state = st.radio(
        label = "8. \tState:",
        options = ["Andhra_Pradesh", "Delhi", "Mizoram", "Punjab", "Sikkim", "Uttarakhand",
                   "West_Bengal", "Haryana", "Tamil_Nadu", "Chhattisgarh", "Chandigarh"],
        index = 0,
        horizontal = True
    )

    # Create the submit button
    submitted = st.form_submit_button("PREDICT")

    # Condition if the input is submitted
    if submitted:
        # Collect the data
        applicant_data_form = {
            "income": person_income,
            "age": person_age,
            "experience": person_experience,
            "current_job_years": current_job_years,
            "current_house_years": current_house_years,
            "married": married,
            "house_ownership": house_ownership,
            "car_ownership": car_ownership,
            "profession": profession,
            "city": city,
            "state": state
        }

        # Create a loading animation to send the data
        with st.spinner("Kirim data untuk diprediksi server ..."):
            res = requests.post("http://localhost:8000/predict",
                                json = applicant_data_form).json()
        # Print the results
        st.write(res)

        st.success(f"""
                Applicant's name: **{app_name}**
                     
                Credit score: **{res['Score']}**  
                Probability of being good: **{res['Proba']}**  
                Recommendation: **{res['Recommendation']}**
            """)