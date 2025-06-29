import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("diabetes.csv")
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
              'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Replace zeros with median for relevant features
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X[cols] = X[cols].replace(0, X[cols].median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, open("diabetes_model.pkl", "wb"))
joblib.dump(scaler, open("diabetes_scaler.pkl", "wb"))
print(f"\n Best model saved: {best_name} with accuracy {best_accuracy:.2f}")
