

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load("titanic_model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
embarked_encoder = joblib.load("embarked_encoder.pkl")


# Function to preprocess the input data
def preprocess_input(input_data):
    """
    Preprocess the input data for prediction.
    Args:
        input_data (dict): Passenger details in JSON format.
    Returns:
        DataFrame: Preprocessed input data.
    """
    # Convert input data to a DataFrame
    df = pd.DataFrame([input_data])

    # Handle missing values
    df["age"] = df.get("age", pd.Series()).fillna(28)  # Replace with typical median value
    df["fare"] = df.get("fare", pd.Series()).fillna(15.0)  # Replace with a reasonable default
    df["embarked"] = df.get("embarked", pd.Series()).fillna("S")  # Replace missing with 'S'

    # Encode categorical variables
    df["sex"] = sex_encoder.transform(df["sex"].map(str))
    df["embarked"] = embarked_encoder.transform(df["embarked"].map(str))

    # Ensure only the required columns are present
    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[feature_columns]

    return df


# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict survival based on passenger details.
    """
    try:
        # Parse input JSON data
        input_data = request.get_json()

        # Preprocess the input data
        processed_data = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return the prediction as a JSON response
        response = {"survived": int(prediction[0])}  # 1 = Survived, 0 = Not survived
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
