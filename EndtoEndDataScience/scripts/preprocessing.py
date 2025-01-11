import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads the Titanic dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def preprocess_data(df):
    """Preprocesses the Titanic dataset."""
    # Clean column names
    df.columns = df.columns.str.strip()

    # Handle missing values
    if 'Age' in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)
    if 'Embarked' in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Drop irrelevant columns
    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "class", "who", "adult_male", "alive", "embark_town", "alone"]
    df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for column in ["Sex", "Embarked"]:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Split into features and target
    if 'Survived' not in df.columns:
        print("Error: 'Survived' column is missing")
        return None, None, None, None, None
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoders
