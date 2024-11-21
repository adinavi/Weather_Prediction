
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    return data

def split_data(data, target_column):
    X = data[['Year', 'Month', 'Day', 'Humidity', 'Rainfall']]
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
