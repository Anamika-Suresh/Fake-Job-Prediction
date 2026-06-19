import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global variables for models and lookup tables
vectorizer = None
clf_log = None
clf_num = None
location_ratio = {}

def load_models():
    global vectorizer, clf_log, clf_num, location_ratio
    
    print("Loading serialized model files...")
    
    # Load CountVectorizer
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    # Load text classifier
    with open('models/clf_log.pkl', 'rb') as f:
        clf_log = pickle.load(f)
        
    # Load numerical classifier
    with open('models/clf_num.pkl', 'rb') as f:
        clf_num = pickle.load(f)
        
    # Load location ratio mapping
    with open('models/location_ratio.json', 'r') as f:
        location_ratio = json.load(f)
        
    print("All models and lookups loaded successfully.")

# Preprocessing logic matching the training pipeline
def preprocess_and_predict(data):
    # Extract fields with default values
    title = data.get('title', '').strip()
    location = data.get('location', '').strip()
    company_profile = data.get('company_profile', '').strip()
    description = data.get('description', '').strip()
    requirements = data.get('requirements', '').strip()
    benefits = data.get('benefits', '').strip()
    telecommuting = int(data.get('telecommuting', 0))
    required_experience = data.get('required_experience', '').strip()
    required_education = data.get('required_education', '').strip()
    industry = data.get('industry', '').strip()
    function = data.get('function', '').strip()
    
    # 1. Parse location to get state and city
    state_city = ""
    ratio = 0.0
    if location:
        parts = [p.strip() for p in location.split(',')]
        # Extract state and city if it's a US location formatted as e.g. "US, NY, New York"
        state = parts[1] if len(parts) > 1 else None
        city = parts[2] if len(parts) > 2 else None
        
        if state and city:
            state_city = f"{state}, {city}"
            # Look up ratio in dictionary
            ratio = location_ratio.get(state_city, 0.0)
            
    # 2. Replicate text concatenation (use spaces for empty fields)
    text_parts = [
        title or " ",
        location or " ",
        company_profile or " ",
        description or " ",
        requirements or " ",
        benefits or " ",
        required_experience or " ",
        required_education or " ",
        industry or " ",
        function or " "
    ]
    text = " ".join(text_parts)
    character_count = len(text)
    
    # 3. Vectorize text features
    text_transformed = vectorizer.transform([text])
    
    # 4. Predict probabilities & labels
    # Note: SGDClassifier with log_loss supports predict_proba
    prob_log = clf_log.predict_proba(text_transformed)[0][1]
    pred_log = clf_log.predict(text_transformed)[0]
    
    # Prepare numerical features
    num_features = np.array([[telecommuting, ratio, character_count]])
    prob_num = clf_num.predict_proba(num_features)[0][1]
    pred_num = clf_num.predict(num_features)[0]
    
    # Combine predictions (logical OR matching the notebook)
    is_fraud = int(pred_log == 1 or pred_num == 1)
    
    return {
        'is_fraud': is_fraud,
        'diagnostics': {
            'text_clf_pred': int(pred_log),
            'text_clf_prob': float(prob_log),
            'num_clf_pred': int(pred_num),
            'num_clf_prob': float(prob_num),
            'computed_ratio': float(ratio),
            'state_city': state_city,
            'character_count': int(character_count)
        }
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept JSON payload
        data = request.get_json(force=True)
        result = preprocess_and_predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load models before starting server
    load_models()
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
