import pandas as pd

def load_csv(file_path='data/raw.csv'):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {file_path}")
        # Convert DataFrame rows to list of dictionaries for downstream processing
        data_records = df.to_dict(orient='records')
        return data_records
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except pd.errors.EmptyDataError:
        print("Empty CSV file.")
        return []
    except pd.errors.ParserError:
        print("Error parsing CSV.")
        return []
    except UnicodeDecodeError:
        print("Encoding error when reading file.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
