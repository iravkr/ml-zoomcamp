from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.bin', 'rb') as f:
   model, dv = pickle.load(f)

app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request
        data = request.get_json(force=True)
        
        # Convert data into DataFrame
        df = pd.DataFrame(data, index=[0])

        # Make prediction
        prediction = model.predict(df)

        # Return prediction
        return jsonify({'EnergyConsumption': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
