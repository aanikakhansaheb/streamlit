import streamlit as st
import pickle
import pandas as pd
import os

# Path to the saved model
file_path = "/Users/sidhanthotchandani/Desktop/BUS 458/salary_model.pkl"
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print("The file is empty.")
    else:
        print(f"The file size is {file_size} bytes.")
else:
    print(f"The file does not exist: {file_path}")

# Load the pre-trained model
with open(file_path, "rb") as file:
    model = pickle.load(file)

# Streamlit app layout
st.markdown(
    "<h1 style='text-align: center; color: red;'>Salary Model</h1>",
    unsafe_allow_html=True
)
st.header("Enter Survey Information To Predict Salary:")

# Define possible inputs
countries = [
    "I do not wish to disclose my location", "Czech Republic", "Romania", "Belgium", 
    "Ireland", "Zimbabwe", "Ecuador", "Hong Kong (S.A.R.)", "Algeria", "Cameroon", 
    "Singapore", "Malaysia", "Nepal", "Sri Lanka", "Ukraine", "Saudi Arabia", 
    "Portugal", "United Arab Emirates", "Ethiopia", "Germany", "Israel", "Ghana", 
    "Philippines", "Netherlands", "South Africa", "Poland", "Chile", 
    "Iran, Islamic Republic of...", "Peru", "Tunisia", "Thailand", "Australia", 
    "Morocco", "Italy", "Kenya", "Argentina", "Viet Nam", "Taiwan", "Bangladesh", 
    "Colombia", "Canada", "Spain", 
    "United Kingdom of Great Britain and Northern Ireland", "France", "South Korea", 
    "Russia", "Turkey", "Indonesia", "Mexico", "Egypt", "China", "Japan", "Pakistan", 
    "Nigeria", "Brazil", "Other", "United States of America", "India"
]

industries = [
    "Academics/Education", "Accounting/Finance", "Broadcasting/Communications", 
    "Computers/Technology", "Energy/Mining", "Government/Public Service", 
    "Insurance/Risk Assessment", "Online Service/Internet-based Services", 
    "Marketing/CRM", "Manufacturing/Fabrication", "Medical/Pharmaceutical", 
    "Non-profit/Service", "Retail/Sales", "Shipping/Transportation", "Other"
]

job_roles = [
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)", 
    "Data Architect", "Data Engineer", "Data Scientist", "Data Administrator", 
    "Developer Advocate", "Machine Learning/ MLops Engineer", 
    "Manager (Program, Project, Operations, Executive-level, etc)", 
    "Research Scientist", "Software Engineer", "Engineer (non-software)", 
    "Statistician", "Teacher / Professor", "Currently not employed", "Other"
]

education_levels = [
    "No formal education past high school", 
    "Some college/university study without earning a bachelor’s degree", 
    "Bachelor’s degree", 
    "Master’s degree", 
    "Doctoral degree", 
    "Professional doctorate", 
    "I prefer not to answer"
]

# User input fields
selected_country = st.selectbox("Choose Your Country", countries)
years_ml = st.slider("Enter # of Years You Have Used Machine Learning Methods:", min_value=0, max_value=20, step=1)
years_code = st.slider("Enter # of Years You Have Been Coding:", min_value=0, max_value=20, step=1)
selected_industry = st.selectbox("In what industry is your current employer/contract (or your most recent employer if retired)?", industries)
selected_job_role = st.selectbox("Select Your Current Job Role (or Most Recent)", job_roles)
age = st.slider("Enter Your Age", min_value=0, max_value=75, step=1)
selected_education = st.selectbox("Select Your Highest Level of Education", education_levels)

# Create a dataframe from input data
input_data = pd.DataFrame({
    "In which country do you currently reside?": [selected_country], 
    "For how many years have you used machine learning methods?": [years_ml],
    "For how many years have you been writing code and/or programming?": [years_code],
    "In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice": [selected_industry],
    "Select the title most similar to your current role (or most recent title if retired): - Selected Choice": [selected_job_role],
    "What is your age (# years)?": [age],
    "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?": [selected_education]
})

# Perform One-Hot Encoding for categorical columns
input_data_encoded = pd.get_dummies(input_data, 
                                    columns=['In which country do you currently reside?', 
                                             'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice', 
                                             'Select the title most similar to your current role (or most recent title if retired): - Selected Choice', 
                                             'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])

# Ensure that the input data columns match the model columns
model_columns = model.feature_names_in_

# Add missing columns (if any) and reorder them
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

input_data_encoded = input_data_encoded[model_columns]

# Make prediction when the button is pressed
if st.button("Predict Salary"):
    prediction = model.predict(input_data_encoded)[0]  # Assuming model.predict returns a scalar
    
    # Display the predicted salary
    st.subheader(f"Predicted Salary: ${prediction:,.2f}")
