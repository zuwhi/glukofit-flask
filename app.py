from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Muat model yang sudah dilatih
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari request
    data = request.get_json()

    # Konversi data ke DataFrame
    df = pd.DataFrame(data)
    
    # Lakukan prediksi dengan model
    predictions = model.predict(df)
    
    # Buat response
    response = {
        'predictions': predictions.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
