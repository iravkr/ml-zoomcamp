from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved XGBoost model
with open("model_xgb.bin", "rb") as model_file:
    model = pickle.load(model_file)

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Ensure the input data matches the model's expected features
        expected_features = ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
                             "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage"]

        for feature in expected_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Prepare the input data as a numpy array
        input_data = np.array([[
            data["N_Days"], data["Age"], data["Bilirubin"], data["Cholesterol"], data["Albumin"],
            data["Copper"], data["Alk_Phos"], data["SGOT"], data["Tryglicerides"], data["Platelets"],
            data["Prothrombin"], data["Stage"]
        ]])

        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        predicted_class = prediction[0]  # Assuming it returns a single class label

        # Map class labels to meaningful Cirrhosis status
        status_mapping = {0: "C", 1: "CL", 2: "D"}
        cirrhosis_status = status_mapping.get(predicted_class, "Unknown")

        # Return the predicted status as JSON
        return jsonify({"status": cirrhosis_status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)  # Run the Flask app on port 9696
