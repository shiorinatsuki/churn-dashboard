import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the saved model and encoders
model = joblib.load('xgboost_model.pkl')
encoders_dict = joblib.load('encoders_dict.pkl')  # Load all categorical encoders
le_target = joblib.load('label_encoder_target.pkl')  # Load target encoder

print("Model and encoders loaded successfully!")

# Load the dataset (using the same data as a test set for simplicity)
df = pd.read_csv('customer_data.csv')

# Drop customerID column (as it's not needed for prediction)
df = df.drop(columns=['customerID'])

# Preprocess the test data (same as in training)
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [col for col in cat_cols if col != 'Churn']  # Exclude 'Churn'

# Use the encoders from encoders_dict for categorical columns
for col in cat_cols:
    le = encoders_dict[col]  # Get the encoder for this column
    df[col] = le.transform(df[col])  # Apply the transformation

# Encode the target variable 'Churn'
df['Churn'] = le_target.transform(df['Churn'])  # Use the target label encoder

# Split the data into features (X_test) and target (y_test)
X_test = df.drop(columns=['Churn'])
y_test = df['Churn']

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, predictions))
