import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), '../models/logreg_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "ML App is running. Send POST requests to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" in JSON payload'}), 400
        
        features = np.array([data['features']])
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

