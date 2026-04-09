from flask import Flask, render_template, request
import joblib  # Use joblib instead of pickle
import numpy as np
import os

app = Flask(__name__)

# --- Load all models and preprocessing tools ---
# We use joblib.load because the files were saved with joblib.dump
try:
    # Check if files exist first to avoid simple errors
    model_path = 'models/milk_model.pkl'
    knn_path = 'models/knn_model.pkl'
    scaler_path = 'models/scaler.pkl'
    le_path = 'models/label_encoder.pkl'

    milk_model = joblib.load(model_path)
    knn_model = joblib.load(knn_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)
    print("All models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    # Initialize variables as None so the app doesn't crash immediately
    milk_model = knn_model = scaler = label_encoder = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if milk_model is None:
        return render_template('index.html', prediction_text="Error: Models not loaded on server.")

    try:
        # 1. Get data from the HTML form
        # Ensure keys match exactly what is in your index.html 'name' attributes
        ph = float(request.form['pH'])
        temp = float(request.form['Temprature'])
        taste = int(request.form['Taste'])
        odor = int(request.form['Odor'])
        fat = int(request.form['Fat'])
        turb = int(request.form['Turbidity'])
        color = int(request.form['Colour'])
        algo = request.form['algorithm']

        # Prepare features for prediction (7 features total)
        features = np.array([[ph, temp, taste, odor, fat, turb, color]])

        # 2. Predict based on selected algorithm
        if algo == 'Decision Tree':
            prediction = milk_model.predict(features)
            
            # If the model outputs a label (string), use it. 
            # If it outputs a number, decode it with LabelEncoder.
            if isinstance(prediction[0], (str, np.str_)):
                result = prediction[0]
            else:
                result = label_encoder.inverse_transform(prediction)[0]
        
        else:
            # KNN Logic: Requires scaling first
            features_scaled = scaler.transform(features)
            prediction = knn_model.predict(features_scaled)
            
            # KNN almost always predicts numbers, so decode it
            if isinstance(prediction[0], (str, np.str_)):
                result = prediction[0]
            else:
                result = label_encoder.inverse_transform(prediction)[0]

        # 3. Return the result to the browser
        return render_template('index.html', 
                               prediction_text=f'Predicted Milk Quality: {result.capitalize()}',
                               algo_used=f'Model used: {algo}')

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)