from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Preprocess the data
    X = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
    X_scaled = scaler.transform(X)

    # Predict the room
    prediction = knn_model.predict(X_scaled)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)