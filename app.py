from flask import Flask, request, jsonify
import joblib
from array import array

app = Flask(__name__)

# Load model and scaler from the same directory as this script
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the request
        features = array('d', data['features'])  # 'd' untuk double (float)

        # Convert array to a list and reshape
        features = [features]
        
        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)

        # Return result
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
