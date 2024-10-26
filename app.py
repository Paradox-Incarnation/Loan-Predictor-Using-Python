import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from groq import Groq

client = Groq(
    api_key=st.secrets["MY_API_KEY"],
)

def generate_suggestions(input_details):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an assistant that provides loan improvement suggestions."},
                {"role": "user", "content": f"Based on the following loan details, provide suggestions to improve loan approval chances:\n{input_details}\nSuggestions:"}
            ],
            max_tokens=1024,
            temperature=1
        )
        suggestions = response.choices[0].message.content
        return suggestions
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return None

df = pd.read_csv("loan_approval_dataset.csv")
df["education"] = df["education"].apply(lambda x: 1 if x == " Graduate" else 0)
df["self_employed"] = df["self_employed"].apply(lambda x: 1 if x == " Yes" else 0)
df["loan_status"] = df["loan_status"].apply(lambda x: 1 if x == " Approved" else 0)

X = df.drop(columns=['loan_status', 'loan_id'], axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    """
    <style>
   .main {
        background-color: #222831; /* Dark background */
        max-width: 600px;
        margin: auto;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: box-shadow 0.3s ease;
    }
    
    .main:hover {
        box-shadow: 0 12px 24px rgba(0,0,0,0.5);
    }

    h1 {
        text-align: center; 
        font-size: 2.5rem; 
        background: linear-gradient(45deg, #f2a365, #00917c, #393e46, #d72323); 
        color:#ffffff;
        margin-bottom: 20px;
    }
    @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
    p {
        text-align: center;
        color: #ffffff;
        font-size: 1.1rem;
    }

    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    .stButton button {
        background-color: #393e46; 
        color: #f2f2f2;
        border: none;
        padding: 12px 28px;
        font-size: 1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: box-shadow 0.4s ease, transform 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 10px #f2a365, 0 0 15px #f2a365; 
        background-color: #f2a365;
        color: #222831; 
    }

    .success {
        color: #f0f0f0;
        background-color: #00917c; 
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 20px;
        animation: fadeIn 1.2s ease;
    }

    .error {
        color: #f0f0f0;
        background-color: #d72323;
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 20px;
        animation: fadeIn 1.2s ease;
    }
    .suggestions {
        color: #f0f0f0; 
        background-color: rgba(255, 108, 108, 0.2);
        padding: 25px; 
        border-radius: 8px; 
        font-size: 1.1rem;
        margin-top: 10px; 
    }

    .suggestions ul {
        list-style-type: disc; 
        padding-left: 25px; 
    }

    .suggestions li {
        margin: 0 0;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True
)

logo_url = "logo.png"  


st.image(logo_url, use_column_width=False, width=200)

st.markdown(
    """
    <style>
    .stImage {
        display: flex;
        justify-content: center; 
    }

    .stImage img {
        border-radius: 20px; 
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1>Loan Predictor: A Tool By <b>AYRA AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p>Enter your details to check loan eligibility.</p>", unsafe_allow_html=True)


no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
income_annum = st.number_input("Annual Income (in INR)", min_value=0, step=100000, value=1000000)
loan_amount = st.number_input("Loan Amount (in INR)", min_value=0, step=100000, value=1000000)
loan_term = st.number_input("Loan Term (in months)", min_value=1, max_value=360, value=12)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=100000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=100000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=100000)

education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0

input_data = np.array([[no_of_dependents, education_encoded, self_employed_encoded, income_annum,
                        loan_amount, loan_term, cibil_score, residential_assets_value,
                        commercial_assets_value, luxury_assets_value, bank_asset_value]])



expected_num_features = X.shape[1]

input_data = input_data.reshape(1, -1)

if st.button("Predict Loan Status"):
    with st.spinner("Analyzing loan details..."):
        time.sleep(2)  
    if input_data.shape[1] != expected_num_features:
        st.error(f"Input data has {input_data.shape[1]} features, expected {expected_num_features}. Please check the inputs.")
    else:
        try:
            input_data_scaled = scaler.transform(input_data) 
            prediction = model.predict(input_data_scaled)[0] 

            if prediction == 1:
                st.markdown('<div class="success">🎉 Congratulations! Your loan has been <b>Approved</b>.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error">✘ Sorry, your loan has been <b>Rejected</b>.</div>', unsafe_allow_html=True)
                input_details = {
                "Number of Dependents": no_of_dependents,
                "Education": education,
                "Self-employed": self_employed,
                "Annual Income": income_annum,
                "Loan Amount": loan_amount,
                "Loan Term": loan_term,
                "CIBIL Score": cibil_score,
                "Residential Assets Value": residential_assets_value,
                "Commercial Assets Value": commercial_assets_value,
                "Luxury Assets Value": luxury_assets_value,
                "Bank Asset Value": bank_asset_value
                }
            
                suggestions = generate_suggestions(input_details)
                if suggestions:
                        st.markdown(f'<div class="suggestions">{suggestions}</div>', unsafe_allow_html=True)


        except ValueError as e:
            st.error(f"Error in scaling input data: {e}. Please check the inputs.")
            st.markdown('<div class="error"> Error in scaling input data. Please check the inputs.</div>', unsafe_allow_html=True)

st.markdown(f" Model Accuracy: {accuracy * 100:.2f}%")
