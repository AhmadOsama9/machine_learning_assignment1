import pandas as pd
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
    
