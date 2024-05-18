import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

def output(x):    
    if x == 0  : return 'normal'
    elif x == 1: return "Sleep Apnea"
    elif x == 2: return "Insomnia" 


# Load the trained model
model = joblib.load("RandomFores.sav")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent in the POST request
    data = request.form
    print ("data is requested ......")
    
   # Extract features from the JSON data
    # Extract features from the form data and convert to numeric types
    features = [
        float(data.get('Age')),
        float(data.get('Sleep Duration')),
        float(data.get('Quality of Sleep')),
        float(data.get('Physical Activity Level')),
        float(data.get('Stress Level')),
        float(data.get('BMI Category')),
        float(data.get('Heart Rate')),
        float(data.get('Daily Steps')),
        float(data.get('systolic_bp')),
        float(data.get('diastolic_bp')),
    ]
    # Convert features to numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)

    # Convert prediction to a regular Python data type
    prediction = int(prediction[0])  # Assuming the prediction is an integer
    
    print ("predict is Finished ......")
    # Return the prediction as JSON response
    return jsonify({'prediction': output(prediction)})


if __name__ == '__main__':
    app.run(port = 18012 , debug=True)