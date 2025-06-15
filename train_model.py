import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('customer_data.csv')

# Drop customerID
df = df.drop(columns=['customerID'])

# Encode categorical columns (excluding target)
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Churn')

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target column separately
le_target = LabelEncoder()
df['Churn'] = le_target.fit_transform(df['Churn'])

# Split data
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Save model
joblib.dump(model, 'xgboost_model.pkl')

# Save encoders
joblib.dump(encoders, 'encoders_dict.pkl')
joblib.dump(le_target, 'label_encoder_target.pkl')

# Save feature names
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("✅ Model and encoders saved successfully!")
