# Loan Predictor: A Tool by AYRA AI  

This repository contains a **machine learning-powered web application** built with **Streamlit** to predict loan approval status. If a loan is rejected, the tool provides **personalized suggestions** using the **Groq API**, helping users improve their chances of approval.

---

## 🔍 Features
- **Loan Status Prediction**: Uses **Logistic Regression** to predict approval status.
- **AI-Generated Suggestions**: When rejected, actionable tips are provided by the Groq API (Llama3-8b model).
- **Interactive UI**: Enter loan-related data such as income, loan term, assets, etc., through an intuitive web interface.
- **Real-time Feedback**: Displays approval status and model accuracy instantly.
- **CSS-enhanced Design**: Clean, modern UI with animations and responsive layout.

---

## 🛠 Technologies Used
- **Python**
- **Streamlit** (for frontend)
- **Pandas & NumPy** (for data handling)
- **Scikit-learn** (for model training)
- **Groq API** (for AI suggestions)
- **HTML/CSS Styling** (for custom UI)

---

## 📂 Project Structure
- **`app.py`**: Main application script.
- **`loan_approval_dataset.csv`**: Dataset used to train the model.
- **`requirements.txt`**: Python dependencies.
- **`logo.png`**: Logo displayed on the app.

---

## ⚙️ Setup and Installation
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-username/loan-predictor-ayra.git
   cd loan-predictor-ayra
   ```
2. **Create and activate a virtual environment**:  
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:  
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Usage
1. Enter details such as **income**, **CIBIL score**, **loan amount**, and **assets**.
2. Click **"Predict Loan Status"** to get the result.
3. If **approved**, you’ll see a success message.
4. If **rejected**, the app provides **suggestions** to improve loan eligibility.

---

## 🔑 API Configuration
Make sure to set the **Groq API key** in the code:
```python
client = Groq(api_key="your_api_key_here")
```

---

## 📊 Model Performance
- **Model Used**: Logistic Regression  
- **Accuracy**: ~80-85%  

---

## ✨ Future Improvements
- Integrate more advanced models (e.g., RandomForest, XGBoost).
- Add more loan-related parameters for better predictions.
- Implement multi-step forms for a better user experience.

---
## 🤝 Contributing
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/your-feature`).  
3. Commit your changes (`git commit -m 'Add feature'`).  
4. Push to the branch (`git push origin feature/your-feature`).  
5. Open a pull request.

---

## 📬 Contact
For questions or feedback, reach out to **[Your Name]** at **[your-email@example.com]**.  

---

This project by **AYRA AI** aims to simplify the loan process and help users improve approval chances through real-time feedback and AI-generated suggestions.