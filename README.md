#Heart Disease Prediction Using Machine Learning
Overview

This project applies machine learning techniques to predict the likelihood of heart disease based on clinical and physiological data.
The dataset was sourced from Kaggle (UCI Heart Disease Data) and processed using Python libraries such as Pandas, Seaborn, and Scikit-learn.

Key Steps

Data Cleaning: Handled missing values, treated outliers, and encoded categorical variables.

Feature Scaling: Normalized numerical features using MinMaxScaler.

Exploratory Analysis: Visualized data through histograms, box plots, and a correlation heatmap.

Modeling: Trained a Decision Tree Classifier to classify patients with or without heart disease.

Evaluation: Measured model performance using accuracy, precision, recall, F1-score, and confusion matrix.

Results
Metric	Score
Accuracy	0.90
Precision (Healthy)	0.96
Recall (Healthy)	0.93
Precision (Disease)	0.15
Recall (Disease)	0.22

The model performs well in identifying non-disease cases but needs improvement in detecting minority (disease) cases due to class imbalance.

Next Steps

Apply SMOTE for balancing the dataset.

Experiment with Random Forest or XGBoost for better generalization.

Perform hyperparameter tuning and cross-validation for optimization.

Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
