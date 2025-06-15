from flask import Flask, render_template, request
import joblib
import pandas as pd
import os  # 👈 Import os for PORT environment variable

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('xgboost_model.pkl')
encoders_dict = joblib.load('encoders_dict.pkl')  # Encoders for categorical features
le_target = joblib.load('label_encoder_target.pkl')  # Encoder for 'Churn'
model_features = joblib.load('model_features.pkl')  # Feature order used in training

@app.route('/')
def home():
    return render_template('index.html')  # Render the main page

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data based on model features
    input_data = {}
    for feature in model_features:
        value = request.form.get(feature)
        input_data[feature] = value

    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data])

    # Apply encoding for categorical features
    for col in df.columns:
        if col in encoders_dict:
            value = df[col][0]
            
            # Check for missing or unseen label
            if value is None or value not in encoders_dict[col].classes_:
                return render_template('index.html', prediction=f"⚠️ Invalid input for '{col}': '{value}'. Please check your input.")
            
            # Safe to transform
            df[col] = encoders_dict[col].transform([value])

    # Make prediction
    prediction = model.predict(df)
    predicted_churn = le_target.inverse_transform(prediction)

    # Render the prediction result and the chart
    return render_template('index.html', prediction=predicted_churn[0], input_data=input_data)

# ✅ Render-compatible Flask run
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Get port from environment
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to public IP
