from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load all models and preprocessing tools
# Ensure these files are inside your 'models' folder
try:
    dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
    knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
except FileNotFoundError:
    print("Error: One or more model files not found in the 'models' folder.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the HTML form
        ph = float(request.form['pH'])
        temp = float(request.form['Temprature'])
        taste = int(request.form['Taste'])
        odor = int(request.form['Odor'])
        fat = int(request.form['Fat'])
        turb = int(request.form['Turbidity'])
        color = int(request.form['Colour'])
        algo = request.form['algorithm']

        # Prepare features for prediction (matching the order of your dataset)
        features = np.array([[ph, temp, taste, odor, fat, turb, color]])

        # 2. Predict based on selected algorithm
        if algo == 'Decision Tree':
            prediction = dt_model.predict(features)
            
            # FIX: Check if DT returned a string (word) or a number
            # If it's a string (like 'low'), we use it directly.
            # If it's a number (like 0), we use the label_encoder.
            if isinstance(prediction[0], (str, np.str_)):
                result = prediction[0]
            else:
                result = label_encoder.inverse_transform(prediction)[0]
        
        else:
            # KNN Logic: Requires scaling first, then predicting numbers
            features_scaled = scaler.transform(features)
            prediction = knn_model.predict(features_scaled)
            
            # KNN predicts numbers, so we must use the label_encoder
            result = label_encoder.inverse_transform(prediction)[0]

        # 3. Return the result to the browser
        return render_template('index.html', 
                               prediction_text=f'Predicted Milk Quality: {result.capitalize()}',
                               algo_used=f'Model used: {algo}')

    except Exception as e:
        # Print the exact error to your VS Code terminal for debugging
        print(f"Prediction Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)