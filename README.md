# 💓 CardioAI — Heart Disease Prediction Web App

CardioAI is an interactive Streamlit-based web application that predicts the likelihood of heart disease using machine learning models. Built for ease of use, it allows users to upload their own health data, select a predictive model, and view diagnostic results with intuitive visualizations.

---

## 🚀 Live Demo

👉 [Launch CardioAI](https://your-streamlit-app-url.streamlit.app)

> Replace this URL with your actual Streamlit app link after deployment.

---

## 🧠 Supported Machine Learning Models

- ✅ Logistic Regression  
- 🌲 Decision Tree  
- 🌳 Random Forest  
- 📍 K-Nearest Neighbors (KNN)  
- 📈 Support Vector Machine (SVM)

Users can choose any of these models via the sidebar in the app.

---

## 📊 Features

- Upload custom CSV datasets or use the built-in heart dataset
- Choose from 5 ML models and see results instantly
- Interactive charts: confusion matrix, classification report
- Fully responsive UI with clean layout

---

## 📁 Dataset Information

The default dataset is derived from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), featuring:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Max Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression
- ...and more

Target column: `1 = heart disease`, `0 = no heart disease`

---

## 📷 Screenshots

| App Homepage | Model Results |
|--------------|----------------|
| ![Home](https://via.placeholder.com/400x250?text=CardioAI+Homepage) | ![Results](https://via.placeholder.com/400x250?text=Prediction+Results) |

> Add real screenshots after deployment for better impact.

---

## 🛠️ Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/cardioai.git
   cd cardioai
```


2. **Install dependencies**
```bash
  pip install -r requirements.txt
  Run the app
```

3. **Run the app**
```bash
  streamlit run app.py
```
