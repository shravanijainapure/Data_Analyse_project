import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Drop useless or single-value columns
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=columns_to_drop)

# 3. Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders to decode data later in the app

# 4. Separate features (X) and target (y)
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Calculate accuracy just to verify
accuracy = model.score(X_test, y_test)
print(print(f"🎉 Model trained successfully with an Accuracy of: {accuracy:.2f}"))

# 7. Save the model and encoders for the Streamlit App
joblib.dump(model, "attrition_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("💾 Artifacts saved: attrition_model.pkl, label_encoders.pkl, feature_columns.pkl")