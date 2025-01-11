import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(file_path):
    """Load and verify the dataset."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()  # Normalize column names
        print(f"Columns in dataset: {df.columns}")
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        return None


def preprocess_data(df):
    """Preprocess the dataset for training."""
    # Ensure required columns exist
    required_columns = {'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'}
    if not required_columns.issubset(set(df.columns)):
        print(f"Error: Required columns are missing. Found: {df.columns}")
        return None, None, None, None, None

    # Handle missing values
    df["age"] = df["age"].fillna(df["age"].median())  # Fill missing 'age' with median
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])  # Fill missing 'embarked' with mode

    # Drop irrelevant columns
    columns_to_drop = [
        "passengerid", "name", "ticket", "cabin", "class", "who", "adult_male",
        "alive", "embark_town", "alone", "deck"
    ]
    df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ["sex", "embarked"]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))  # Encode as numeric
        label_encoders[column] = le

    # Split features and target
    X = df.drop("survived", axis=1)
    y = df["survived"]
    print(f"Processed feature columns: {X.columns}")
    print(f"Target column: survived")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoders


def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")


def save_model_and_encoders(model, label_encoders):
    """Save the trained model and encoders to disk."""
    joblib.dump(model, "titanic_model.pkl")
    for column, encoder in label_encoders.items():
        joblib.dump(encoder, f"{column}_encoder.pkl")
    print("Model and encoders saved successfully!")


if __name__ == "__main__":
    file_path = "/home/gnagaraj/PycharmProjects/EndtoEndDataScience/data/titanic.csv"

    # Load the data
    data = load_data(file_path)
    if data is None:
        exit()

    # Preprocess the data
    X_train, X_test, y_train, y_test, encoders = preprocess_data(data)
    if X_train is None or y_train is None:
        exit()

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model and encoders
    save_model_and_encoders(model, encoders)

