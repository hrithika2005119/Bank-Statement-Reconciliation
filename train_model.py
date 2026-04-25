import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your company dataset
df = pd.read_csv("company_transactions_600_rows.csv")

# Encode categorical data
le_desc = LabelEncoder()
le_type = LabelEncoder()

df['Description'] = le_desc.fit_transform(df['Description'])
df['Type'] = le_type.fit_transform(df['Type'])

X = df[['Description', 'Amount', 'Type']]
y = df['Fraud_Label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "fraud_model.pkl")
joblib.dump(le_desc, "desc_encoder.pkl")
joblib.dump(le_type, "type_encoder.pkl")

print("✅ Model saved successfully!")