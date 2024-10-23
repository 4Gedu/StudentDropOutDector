#compare metrics XPBoost and DropOutModelAI Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset (replace with your own dataset)
# For example purposes, we assume 'data.csv' contains your features and 'target' column for dropout status
data = pd.read_csv('data.csv')

# Separate features and target
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target (dropout status)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Train the models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# Define a function to calculate and display performance metrics
def evaluate_model(name, y_true, y_pred):
    print(f"Performance Metrics for {name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("-" * 40)

# Evaluate Random Forest
evaluate_model("Random Forest", y_test, rf_preds)

# Evaluate XGBoost
evaluate_model("XGBoost", y_test, xgb_preds)

# Print detailed classification report
print("\nDetailed Classification Report:")
print(f"Random Forest:\n{classification_report(y_test, rf_preds)}")
print(f"XGBoost:\n{classification_report(y_test, xgb_preds)}")
