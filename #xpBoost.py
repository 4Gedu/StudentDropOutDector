#xpBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assume we have the previous data collection and feature engineering functions


# 1. Data Collection
def collect_data():
    np.random.seed(42)
    n_students = 1000
    data = {
        'student_id': range(1, n_students + 1),
        'gpa': np.random.uniform(1.0, 4.0, n_students),
        'classes_attended': np.random.randint(0, 100, n_students),
        'total_classes': np.full(n_students, 100),
        'assignments_completed': np.random.randint(0, 50, n_students),
        'total_assignments': np.full(n_students, 50),
        'latest_test_score': np.random.uniform(0, 100, n_students),
        'average_test_score': np.random.uniform(50, 90, n_students),
        'recent_attendance': np.random.randint(0, 10, n_students),
        'average_attendance': np.random.uniform(5, 10, n_students),
        'recent_assignment_completion': np.random.randint(0, 5, n_students),
        'average_assignment_completion': np.random.uniform(2, 5, n_students),
        'tuition_balance': np.random.uniform(0, 10000, n_students),
        'financial_aid': np.random.uniform(0, 15000, n_students),
        'part_time_job_hours': np.random.randint(0, 30, n_students),
        'missed_payments': np.random.randint(0, 3, n_students),
        'scholarship_amount': np.random.uniform(0, 5000, n_students),
        'dropout': np.random.choice([0, 1], n_students, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# 2. Feature Engineering
def engineer_features(df):
    # Academic features
    df['attendance_rate'] = df['classes_attended'] / df['total_classes']
    df['assignment_completion_rate'] = df['assignments_completed'] / df['total_assignments']
    df['recent_test_performance'] = df['latest_test_score'] / df['average_test_score']
    df['recent_attendance_change'] = df['recent_attendance'] - df['average_attendance']
    df['recent_assignment_change'] = df['recent_assignment_completion'] - df['average_assignment_completion']
    
    # Financial features
    df['net_tuition'] = df['tuition_balance'] - df['financial_aid'] - df['scholarship_amount']
    df['financial_stress_score'] = (df['net_tuition'] / 10000) + (df['missed_payments'] * 0.5) - (df['scholarship_amount'] / 5000)
    df['work_study_balance'] = df['part_time_job_hours'] / df['total_classes']
    
    # Detect sudden changes
    sudden_changes = detect_sudden_changes(df[['attendance_rate', 'assignment_completion_rate', 'recent_test_performance', 'financial_stress_score']])
    df['sudden_change_flag'] = sudden_changes.astype(int)
    
    return df

# 3. Data Preprocessing

def preprocess_data(df):
    X = df.drop(['dropout', 'student_id'], axis=1)
    y = df['dropout']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {'Accuracy': accuracy, 'AUC': auc}
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        print("\n")
    
    return results

def plot_model_comparison(results):
    models = list(results.keys())
    accuracies = [result['Accuracy'] for result in results.values()]
    aucs = [result['AUC'] for result in results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy')
    ax.bar([i + width/2 for i in x], aucs, width, label='AUC')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def tune_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    
    return grid_search.best_estimator_

def plot_feature_importance(model, X_train):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(pos, feature_importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(X_train.columns[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for XGBoost Model')
    plt.tight_layout()
    plt.show()

def main():
    # Assume we have collected and engineered the data
    df = collect_data()
    df = engineer_features(df)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Compare models
    results = compare_models(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)
    
    # Tune XGBoost
    best_xgb = tune_xgboost(X_train, y_train)
    
    # Evaluate tuned XGBoost
    y_pred = best_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])
    
    print("Tuned XGBoost Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    plot_feature_importance(best_xgb, X_train)

if __name__ == "__main__":
    main()