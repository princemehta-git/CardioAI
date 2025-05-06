import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="CardioAI ~ Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>CardioAI ~ ❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app uses machine learning to predict heart disease based on patient data.")

# Sidebar
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["K-Nearest Neighbors (KNN)", 
     "Support Vector Machine (SVM)", 
     "Decision Tree", 
     "Random Forest", 
     "Gaussian Naive Bayes", 
     "Logistic Regression",
     "Neural Network (ANN)"]
)

# Advanced model parameters
with st.sidebar.expander("Advanced Model Parameters"):
    if model_choice == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Number of neighbors (K)", 1, 20, 5)
    elif model_choice == "Support Vector Machine (SVM)":
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
    elif model_choice == "Decision Tree":
        max_depth = st.slider("Maximum depth", 1, 20, 5)
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth_rf = st.slider("Maximum depth of trees", 1, 20, 5)
    elif model_choice == "Neural Network (ANN)":
        epochs = st.slider("Epochs", 10, 100, 50)
        batch_size = st.slider("Batch Size", 8, 64, 16)
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

# Data upload
st.sidebar.title("Data Options")
upload_option = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV", "Use default dataset"]
)

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            df = None
    else:
        st.sidebar.info("Please upload a CSV file")
        df = None
else:
    # Use default dataset
    try:
        df = pd.read_csv("Heart.csv")
        if df is not None:
            st.sidebar.success("✅ Default dataset loaded!")
    except FileNotFoundError:
        st.sidebar.error("Default dataset not found. Please upload a CSV file.")
        df = None

# Function to preprocess data
def preprocess_data(dataframe):
    # Make a copy to avoid modifying the original
    df_processed = dataframe.copy()
    
    # Drop unnecessary columns if they exist
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop('Unnamed: 0', axis=1)
    
    # Handle the target variable
    if 'target' not in df_processed.columns:
        if 'thalach' in df_processed.columns and df_processed['thalach'].dtype == object:
            # Convert Yes/No to binary
            if set(df_processed['thalach'].unique()).issubset({'Yes', 'No'}):
                df_processed['target'] = df_processed['thalach'].map({'Yes': 1, 'No': 0})
                df_processed = df_processed.drop('thalach', axis=1)
        else:
            st.error("Target column not found. Please ensure your dataset has a 'target' column.")
            return None
    
    # One-hot encode categorical variables
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=False)
    
    # Identify numerical columns for scaling
    numerical_columns = ['age', 'trestbps', 'chol', 'oldpeak']
    if 'thalach' in df_processed.columns and df_processed['thalach'].dtype != object:
        numerical_columns.append('thalach')
    
    # Return the processed dataframe and numerical columns
    return df_processed, numerical_columns

# Function to create and train models
def train_model(X_train, X_test, y_train, y_test, model_name):
    if model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "Support Vector Machine (SVM)":
        model = SVC(kernel=kernel, C=C, probability=True)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
    elif model_name == "Gaussian Naive Bayes":
        model = GaussianNB()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Neural Network (ANN)":
        # For ANN, we'll return a different object
        return train_neural_network(X_train, X_test, y_train, y_test)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, y_pred, accuracy, cm, report

def train_neural_network(X_train, X_test, y_train, y_test):
    # Convert to one-hot encoding for neural network
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # Create model
    model = Sequential()
    model.add(Dense(16, 
                   input_dim=X_train.shape[1], 
                   kernel_initializer='normal',
                   kernel_regularizer=regularizers.l2(0.01),
                   activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, 
                   kernel_initializer='normal',
                   kernel_regularizer=regularizers.l2(0.01),
                   activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])
    
    # Train model with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {int(progress * 100)}%")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[CustomCallback()]
    )
    
    # Clear progress display
    status_text.empty()
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, y_pred, accuracy, cm, report, history

# Main content
if df is not None:
    # Data exploration
    with st.expander("Data Exploration", expanded=False):
        st.subheader("Dataset Overview")
        st.write(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Shape:", df.shape)
        with col2:
            st.write("Dataset Info:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        st.write("Dataset Description:")
        st.write(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.warning("Missing values detected in the dataset:")
            st.write(missing_values[missing_values > 0])
    
    # Preprocess data
    processed_df, numerical_columns = preprocess_data(df)
    
    if processed_df is not None:
        # Split data
        X = processed_df.drop('target', axis=1)
        y = processed_df['target']
        
        # Standardize features
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        
        # Train-test split
        test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        st.markdown(f"<h2 class='sub-header'>Training {model_choice} Model</h2>", unsafe_allow_html=True)
        
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Neural Network (ANN)":
                model, y_pred, accuracy, cm, report, history = train_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
            else:
                model, y_pred, accuracy, cm, report = train_model(
                    X_train, X_test, y_train, y_test, model_choice
                )
        
        # Display results
        st.markdown(f"<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        
        # Metrics in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Precision (Class 1)", f"{report['1']['precision']:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Recall (Class 1)", f"{report['1']['recall']:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Confusion Matrix and Classification Report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {model_choice}")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            # Format the report for better readability
            for col in ['precision', 'recall', 'f1-score']:
                report_df[col] = report_df[col].apply(lambda x: f"{x:.2%}")
            st.dataframe(report_df)
        
        # Model specific visualizations
        st.markdown(f"<h2 class='sub-header'>Model Insights</h2>", unsafe_allow_html=True)
        
        if model_choice == "K-Nearest Neighbors (KNN)":
            # KNN - Accuracy vs K value
            st.subheader("KNN Performance with Different K Values")
            k_range = range(1, 11)
            k_scores = []
            
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                k_scores.append(knn.score(X_test, y_test))
            
            fig = px.line(
                x=list(k_range), y=k_scores,
                labels={'x': 'Number of Neighbors (K)', 'y': 'Accuracy'},
                title='KNN Accuracy for Different K Values',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "Support Vector Machine (SVM)":
            # SVM - Different kernels comparison
            st.subheader("SVM Performance with Different Kernels")
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            kernel_scores = []
            
            for k in kernels:
                svm = SVC(kernel=k)
                svm.fit(X_train, y_train)
                kernel_scores.append(svm.score(X_test, y_test))
            
            fig = px.bar(
                x=kernels, y=kernel_scores,
                labels={'x': 'Kernel Type', 'y': 'Accuracy'},
                title='SVM Accuracy for Different Kernels',
                color=kernel_scores
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "Decision Tree" or model_choice == "Random Forest":
            # Feature importance
            if model_choice == "Decision Tree":
                importances = model.feature_importances_
                feature_names = X.columns
            else:  # Random Forest
                importances = model.feature_importances_
                feature_names = X.columns
            
            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]
            
            # Rearrange feature names so they match the sorted feature importances
            names = [feature_names[i] for i in indices]
            
            # Create plot
            fig = px.bar(
                x=importances[indices][:10], y=names[:10],
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title=f'Top 10 Feature Importances - {model_choice}',
                color=importances[indices][:10]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "Logistic Regression":
            # Coefficients
            if hasattr(model, 'coef_'):
                coef = model.coef_.flatten()
                feature_names = X.columns
                
                # Create a DataFrame for better visualization
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coef
                })
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                fig = px.bar(
                    coef_df.head(10),
                    x='Coefficient', y='Feature',
                    orientation='h',
                    title='Top 10 Logistic Regression Coefficients',
                    color='Coefficient'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif model_choice == "Neural Network (ANN)":
            # Training history
            st.subheader("Neural Network Training History")
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(
                go.Scatter(y=history.history['accuracy'], name="Train Accuracy")
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_accuracy'], name="Validation Accuracy")
            )
            
            # Add figure layout
            fig.update_layout(
                title="Training and Validation Accuracy",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                legend_title="Metrics",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss plot
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(y=history.history['loss'], name="Train Loss")
            )
            fig2.add_trace(
                go.Scatter(y=history.history['val_loss'], name="Validation Loss")
            )
            
            fig2.update_layout(
                title="Training and Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend_title="Metrics",
                template="plotly_white"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Prediction on new data
        st.markdown(f"<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
        
        with st.expander("Predict on New Data"):
            st.write("Enter patient information to predict heart disease risk:")
            
            # Create input fields based on original dataset columns
            input_data = {}
            
            # Get original column names before one-hot encoding
            original_columns = set(df.columns) - {'target'}
            if 'thalach' in original_columns and df['thalach'].dtype == object:
                original_columns.remove('thalach')
            
            # Create a form for user input
            with st.form("prediction_form"):
                # Create columns for a cleaner layout
                col1, col2 = st.columns(2)
                
                for col in sorted(original_columns):
                    # Check if column is categorical
                    if col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                        # Get unique values for categorical columns
                        unique_values = df[col].unique()
                        
                        # Create a selectbox for categorical variables
                        if col == 'sex':
                            options = {0: 'Female', 1: 'Male'} if set(unique_values) == {0, 1} else unique_values
                            input_data[col] = col1.selectbox(f"{col.capitalize()}", options=options)
                        else:
                            input_data[col] = col1.selectbox(f"{col.capitalize()}", options=unique_values)
                    else:
                        # Create a number input for numerical variables
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        default_val = float(df[col].mean())
                        
                        input_data[col] = col2.number_input(
                            f"{col.capitalize()}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=1.0 if col in ['age'] else 0.1
                        )
                
                # Submit button
                submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Prepare input data for prediction
                input_df = pd.DataFrame([input_data])
                
                # Apply the same preprocessing as the training data
                processed_input, _ = preprocess_data(input_df)
                
                # Ensure all columns match the training data
                for col in X.columns:
                    if col not in processed_input.columns:
                        processed_input[col] = 0
                
                # Reorder columns to match training data
                processed_input = processed_input[X.columns]
                
                # Scale numerical features
                processed_input[numerical_columns] = scaler.transform(processed_input[numerical_columns])
                
                # Make prediction
                if model_choice == "Neural Network (ANN)":
                    prediction_prob = model.predict(processed_input)
                    prediction = np.argmax(prediction_prob, axis=1)[0]
                    probability = prediction_prob[0][prediction]
                else:
                    prediction = model.predict(processed_input)[0]
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(processed_input)[0][prediction]
                    else:
                        probability = None
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.error(f"❌ Heart Disease Detected (Confidence: {probability:.2%})")
                else:
                    st.success(f"✅ No Heart Disease Detected (Confidence: {probability:.2%})")
        
        # Download trained model
        st.markdown(f"<h2 class='sub-header'>Download Model</h2>", unsafe_allow_html=True)
        
        if model_choice != "Neural Network (ANN)":
            # Save model using pickle
            import pickle
            model_pickle = pickle.dumps(model)
            
            # Create download button
            st.download_button(
                label="Download Trained Model",
                data=model_pickle,
                file_name=f"heart_disease_{model_choice.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )
        else:
            st.info("Neural Network models require TensorFlow to be loaded. Please use the export functionality in TensorFlow to save this model.")
else:
    st.warning("Please upload a dataset or use the default dataset to proceed.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ By Prince Mehta")