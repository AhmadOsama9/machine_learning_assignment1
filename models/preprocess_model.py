import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Our Target should be the CO2 emission
# there is multiple columns in our dataset but if we assume one target then it will be it.

# Separate features and targets
def separate_features_and_target(data, target_column):
    features = data.drop(columns=[target_column]) # we drop the target which will be CO2 emission in regression and Emission class in classification
    target = data[target_column]
    return features, target 
    
# label encoder
def encode_categorical_features_targets_using_LE(features, target):

    le = LabelEncoder()

    categorical_columns = features.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        features[col] = le.fit_transform(features[col])

    if isinstance(target, pd.DataFrame):
        target_columns = target.select_dtypes(include=['object']).columns
        for col in target_columns:
            target[col] = le.fit_transform(target[col])
    elif target.dtype == 'object':
        target = le.fit_transform(target)
    
    return features, target



def shuffle_and_split_data(features, target1, target2, test_size=0.2, random=42):
    X_train, X_test, y_train1, y_test1, y_train2, y_test2 =  train_test_split(features, target1, target2, test_size=test_size, random_state=random)
    return X_train, X_test, y_train1, y_test1, y_train2, y_test2


def numeric_scale_test_train_data(data, numeric_features_columns, categorical_features_columns):
    scaler = StandardScaler()

    data_n = data[numeric_features_columns]

    scaled_data = scaler.fit_transform(data_n)
    
    scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_features_columns, index=data.index)
   
    
    categorical_data = data[categorical_features_columns]

    final_data = pd.concat([scaled_data_df, categorical_data], axis=1)
    return final_data   