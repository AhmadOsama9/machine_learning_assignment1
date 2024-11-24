import pandas as pd
import seaborn as sns # an abbrivation from a drama so as a meme NICE
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data, None
    except Exception as e:  
        return None, str(e)


