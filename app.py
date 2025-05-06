import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("💓 Heart Disease Prediction App")

# Sidebar
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model", 
    ("Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"))

# Upload CSV
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load default data if nothing is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    df = pd.read_csv("Heart.csv")
    st.info("ℹ️ Using default heart dataset.")

# Show data
if st.checkbox("Show Raw Data"):
    st.subheader("Dataset")
    st.write(df)

# Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and Predict
if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "KNN":
    model = KNeighborsClassifier()
elif model_choice == "SVM":
    model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show Metrics
st.subheader(f"📊 Results - {model_choice}")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.write("**Classification Report**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
