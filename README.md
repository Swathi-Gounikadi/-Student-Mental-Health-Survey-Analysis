# 🧠 Student Mental Health Prediction using Machine Learning

An end-to-end Machine Learning project for predicting student depression levels using psychological, academic, and lifestyle-related factors.

---

## 📌 Project Overview

Student mental health has become a major concern due to increasing academic pressure, stress, sleep disorders, and lifestyle imbalance.

This project aims to build a Machine Learning system that can:

- Predict whether a student is depressed or not
- Analyze mental health-related behavioral patterns
- Identify major contributing factors affecting depression
- Help institutions and counselors detect at-risk students early

The project uses multiple ML algorithms and compares their performance to build the best prediction model.

---

## 🎯 Problem Statement

Students often experience:

- Academic pressure
- Anxiety and stress
- Poor sleep habits
- Financial stress
- Lack of social support

These factors negatively impact mental wellness.

The goal of this project is to classify students into:

- Healthy → 0
- Depressed → 1

using survey-based psychological and lifestyle data.

---

## 📂 Dataset Features

The dataset contains various features related to:

### 👤 Personal Information
- Gender
- Age
- City
- Degree
- Profession

### 📚 Academic Factors
- Academic Pressure
- Study Satisfaction
- CGPA
- Work/Study Hours

### 🧠 Psychological Factors
- Suicidal Thoughts
- Financial Stress
- Family History of Mental Illness

### 🍔 Lifestyle Factors
- Sleep Duration
- Dietary Habits

### 🎯 Target Variable
- Depression (0 or 1)

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

---

# ⚙️ Machine Learning Workflow

## 1️⃣ Data Loading
- Loaded dataset using Pandas
- Checked shape and dataset information

## 2️⃣ Data Cleaning
- Removed unnecessary columns
- Checked missing values
- Removed duplicates

## 3️⃣ Outlier Handling
Used IQR method to handle outliers in:
- Age
- CGPA
- Work/Study Hours

## 4️⃣ Exploratory Data Analysis (EDA)
Performed:
- Count plots
- Histograms
- Correlation heatmaps
- Distribution analysis

### Key Insights
- Poor sleep is strongly linked to depression
- Academic pressure highly affects mental health
- Study satisfaction negatively correlates with depression
- Financial stress contributes significantly

---

# 🔄 Data Preprocessing

## Feature Scaling
Used:
- StandardScaler for numerical features

## Encoding
Used:
- OrdinalEncoder for categorical features

---

# ⚖️ Handling Imbalanced Data

Applied:
SMOTE (Synthetic Minority Oversampling Technique)

to balance depression classes.

---

# 🧪 Feature Selection

Used:
mutual_info_classif

to identify important features.

### Top Important Features
- Suicidal Thoughts
- Academic Pressure
- Financial Stress
- Sleep Duration
- Study Satisfaction

---

# 🤖 Models Implemented

The following Machine Learning models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- AdaBoost
- Naive Bayes
- XGBoost

---

# 🚀 Best Performing Model

## ✅ XGBoost Classifier

XGBoost achieved the best overall performance after hyperparameter tuning.

### Hyperparameter Tuning
Used:
GridSearchCV

for optimization.

---

# 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 📈 Key Findings

- Psychological factors are the strongest predictors of depression
- Sleep quality significantly impacts mental health
- Academic pressure increases depression risk
- Lifestyle habits also influence emotional wellness

---

# 💾 Model Saving

Saved:
- Preprocessor
- Feature Names
- Selected Features
- Trained XGBoost Model

Using:
pickle

---

# 📁 Project Structure

```bash
├── student_depression_dataset.csv
├── Student_Mental_Health_Analysis.ipynb
├── preprocessor.pkl
├── top_features.pkl
├── feature_names.pkl
├── xgb_model.pkl
└── README.md


🔮 Future Improvements
Deploy as a web application using Streamlit
Add Deep Learning models
Build real-time student mental health dashboard
Integrate chatbot-based mental health support
Improve interpretability using SHAP values

📚 Conclusion

This project demonstrates how Machine Learning can help analyze and predict student depression using psychological, academic, and lifestyle data.

The system can assist educational institutions in identifying vulnerable students early and providing timely support.
