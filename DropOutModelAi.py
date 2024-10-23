import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# 4. Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, conf_matrix, class_report

# 6. Prediction Function
def predict_dropout_risk(model, student_data, scaler):
    student_data_scaled = scaler.transform(student_data)
    prediction = model.predict(student_data_scaled)
    probability = model.predict_proba(student_data_scaled)[:, 1]
    return prediction, probability

# 7. Sudden Change Detection
def detect_sudden_changes(student_data, threshold=0.5):
    diffs = student_data.diff().abs()
    sudden_changes = (diffs > threshold).any(axis=1)
    return sudden_changes

# 8. Real-time Monitoring
def monitor_student_activity(student_id, new_data, model, scaler):
    new_data_scaled = scaler.transform(new_data)
    dropout_risk = model.predict_proba(new_data_scaled)[0][1]
    sudden_changes = detect_sudden_changes(new_data)
    
    financial_stress = new_data['financial_stress_score'].values[0]
    risk_threshold = 0.7 - (financial_stress * 0.05)  # Lower threshold for financially stressed students
    
    if sudden_changes.any() or dropout_risk > risk_threshold:
        alert_counselor(student_id, dropout_risk, sudden_changes, financial_stress)
    
    return dropout_risk, sudden_changes, financial_stress

def alert_counselor(student_id, dropout_risk, sudden_changes, financial_stress):
    print(f"ALERT: Student {student_id} at high risk of dropout.")
    print(f"Dropout risk: {dropout_risk:.2f}")
    print(f"Sudden changes detected: {sudden_changes.to_dict()}")
    print(f"Financial stress score: {financial_stress:.2f}")
    # In a real system, this would send an alert to the counselor and financial aid office

# Main function to run the entire process
def main():
    # Collect and prepare data
    df = collect_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    
    print(f"Model Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
    # Simulate real-time monitoring for a few students
    for student_id in range(1, 6):
        new_data = pd.DataFrame({
            'gpa': [np.random.uniform(1.0, 4.0)],
            'classes_attended': [np.random.randint(0, 100)],
            'total_classes': [100],
            'assignments_completed': [np.random.randint(0, 50)],
            'total_assignments': [50],
            'latest_test_score': [np.random.uniform(0, 100)],
            'average_test_score': [np.random.uniform(50, 90)],
            'recent_attendance': [np.random.randint(0, 10)],
            'average_attendance': [np.random.uniform(5, 10)],
            'recent_assignment_completion': [np.random.randint(0, 5)],
            'average_assignment_completion': [np.random.uniform(2, 5)],
            'tuition_balance': [np.random.uniform(0, 10000)],
            'financial_aid': [np.random.uniform(0, 15000)],
            'part_time_job_hours': [np.random.randint(0, 30)],
            'missed_payments': [np.random.randint(0, 3)],
            'scholarship_amount': [np.random.uniform(0, 5000)]
        })
        new_data = engineer_features(new_data)
        
        dropout_risk, sudden_changes, financial_stress = monitor_student_activity(student_id, new_data, model, scaler)
        print(f"\nStudent {student_id}:")
        print(f"Dropout Risk: {dropout_risk:.2f}")
        print(f"Sudden Changes: {sudden_changes.to_dict()}")
        print(f"Financial Stress Score: {financial_stress:.2f}")

if __name__ == "__main__":
    main()