import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Our Target should be the CO2 emission
# there is multiple columns in our dataset but if we assume one target then it will be it.

# Separate features and targets
def separate_features_and_target(data, target_column):
    features = data.drop(columns=[target_column]) # we drop the target which will be CO2 emission in regression and Emission class in classification
    target = data[target_column]
    return features, target 


# categorical features and targets are encoded
# so here I also think we will use it for both regression and classification
def encode_categorical_features_targets(features, target):
    categorical_columns = features.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) == 0:
        return features, target
    
    features = pd.get_dummies(features, columns=categorical_columns)

    if (target.dtype == 'object'):
        target = pd.get_dummies(target) # It will encode it using one-hot encoding where we will have a series of binary columns
        
    return features, target
    
# label encoder
def encode_categorical_features_targets_using_LE(features, target):

    le = LabelEncoder()

    categorical_columns = features.select_dtypes(include=['object']).columns
    target_columns = target.select_dtypes(include=['object']).columns

    if len(categorical_columns) == 0:
        return features, target
    
    for col in categorical_columns:
        features[col] = le.fit_transform(features[col])

    if (target.dtype == 'object'):
        target_columns = le.fit_transform(target_columns)

    return features, target



def shuffle_and_split_data(features, target, test_size=0.2, random=42):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random)
    return X_train, X_test, y_train, y_test


def numeric_scale_test_train_data(data):
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    return scaled_data