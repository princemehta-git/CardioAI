# 💓 CardioAI — Heart Disease Prediction Web App

CardioAI is an interactive Streamlit-based web application that predicts the likelihood of heart disease using machine learning models. Built for healthcare professionals and individuals interested in heart health, it allows users to upload their own health data, select from multiple predictive models, and view diagnostic results with intuitive visualizations.

---

## 🚀 Live Demo

👉 [Launch CardioAI](https://cardioai-app.streamlit.app/)

---

## 🧠 Supported Machine Learning Models

- ✅ Logistic Regression - Classic statistical approach for binary classification
- 🌲 Decision Tree - Rule-based model with transparent decision paths
- 🌳 Random Forest - Ensemble method combining multiple decision trees
- 📍 K-Nearest Neighbors (KNN) - Classification based on feature similarity
- 📊 Support Vector Machine (SVM) - Powerful classifier with different kernel options
- 📈 Gaussian Naive Bayes - Probabilistic classifier based on Bayes' theorem
- 🧪 Neural Network (ANN) - Deep learning approach with customizable architecture

Users can choose any of these models via the sidebar and adjust advanced parameters for optimal performance.

---

## 📊 Features

- **Data Flexibility**: Upload custom CSV datasets or use the built-in heart disease dataset
- **Model Selection**: Choose from 7 different ML models with customizable parameters
- **Interactive Visualizations**: Confusion matrix, classification reports, and model-specific insights
- **Real-time Predictions**: Input patient data and receive instant heart disease risk assessment
- **Model Performance Analysis**: Compare accuracy, precision, recall across different models
- **Advanced Customization**: Fine-tune model parameters through an intuitive interface
- **Downloadable Models**: Export trained models for external use
- **Responsive Design**: Clean, intuitive UI that works on desktop and mobile devices

---

## 📁 Dataset Information

The default dataset is derived from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), featuring:

| Feature | Description |
|---------|-------------|
| Age | Age in years |
| Sex | Gender (1 = male, 0 = female) |
| CP | Chest pain type (0-3) |
| Trestbps | Resting blood pressure (mm Hg) |
| Chol | Serum cholesterol (mg/dl) |
| FBS | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| RestECG | Resting electrocardiographic results |
| Thalach | Maximum heart rate achieved |
| Exang | Exercise induced angina (1 = yes, 0 = no) |
| Oldpeak | ST depression induced by exercise relative to rest |
| Slope | Slope of the peak exercise ST segment |
| CA | Number of major vessels colored by fluoroscopy (0-3) |
| Thal | Thalassemia (normal, fixed defect, reversible defect) |

Target column: `1 = heart disease present`, `0 = no heart disease`

## 🛠️ Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/cardioai.git
   cd cardioai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Run the app

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
