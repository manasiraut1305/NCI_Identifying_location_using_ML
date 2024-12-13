from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import joblib

# Initialize the global DataFrame
final = pd.DataFrame(columns=['latitude', 'longitude', 'room'])

app = Flask(__name__, template_folder="templates")

# Load the trained model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route("/")
def hello():
    return render_template('index.html')

location_data = dict()

@app.route('/process', methods=['POST'])
def process():
     global final  # Reference the global DataFrame
     data = request.get_json()
     latitude = data.get('latitude')
     longitude = data.get('longitude')
     room = data.get('room')
     location_data = {
         'latitude': latitude,
         'longitude': longitude,
         'room': room
     }

    # Create a DataFrame from the received data
     df = pd.DataFrame([location_data])
    
     #Append to the global DataFrame
     final = final._append(df, ignore_index=True)

      #Save the DataFrame to CSV
     final.to_csv('location.csv', index=False)

     try:
         # Append the new data to the JSON file
         with open('location.json', 'a') as f:
             json.dump(location_data, f, indent=2)
             f.write('\n')  # Add a new line for each entry
         return jsonify({'message': 'Location saved successfully', 'data': location_data})
     except Exception as e:
         print(f"Error writing to file: {e}")
         return jsonify({'message': 'Internal Server Error'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    room = data.get('room')
    
    # Preprocess the data
    X = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
    X_scaled = scaler.transform(X)

    # Predict the room
    prediction = knn_model.predict(X_scaled)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
