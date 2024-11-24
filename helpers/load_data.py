import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data, None
    except Exception as e:  
        return None, str(e)