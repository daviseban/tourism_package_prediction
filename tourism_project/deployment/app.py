import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="daviseban/tourism-package-prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts the package adoption by an individual based on certain params.
""")

# Customer Details
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=20, value=1, step=1)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=2, step=1)
passport = st.selectbox("Passport", ["No", "Yes"])
own_car = st.selectbox("Own Car", ["No", "Yes"])
number_of_children_visiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Others"])
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000, step=1000)

# Customer Interaction Data
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Premium", "Others"])
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=1, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=30, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfFollowups': number_of_followups
}])

# Predict button
if st.button("Predict Package Adoption"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Predicted adoption: **${prediction} **")
