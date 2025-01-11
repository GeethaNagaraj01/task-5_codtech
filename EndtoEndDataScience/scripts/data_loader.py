import pandas as pd

def loadData(file_path):
    """Loads the dataset from the given file path."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file.")
        return None

if __name__ == "__main__":
    file_path = "/home/gnagaraj/PycharmProjects/EndtoEndDataScience/data/titanic.csv"
    data = loadData(file_path)
    if data is not None:
        print(data.head())  # Displaying first 5 rows of the dataset
