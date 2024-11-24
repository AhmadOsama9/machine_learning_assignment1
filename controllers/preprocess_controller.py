from helpers.load_data import load_data
import models.preprocess_model as preprocess
from views.display_view import print_data_error, print_data_success


def handle_preprocess():
    file_path = 'data/co2_emissions_data.csv'
    
    data, error = load_data(file_path)
    
    if data is not None:
        numeric_features = data.select_dtypes(include=['int64', 'float64'])
        numeric_features_columns = numeric_features.columns
        categorical_features = data.select_dtypes(exclude=['int64', 'float64'])
        categorical_features_columns = categorical_features.columns
        
        # So we will use the functions according to the target and features
        # depending on classification or regression I think
        features, target = preprocess.separate_features_and_target(data, 'Emission Class')
        # features, target = preprocess.encode_categorical_features_targets(features, target)
        features, target = preprocess.encode_categorical_features_targets_using_LE(features, target)
        X_train, X_test, y_train, y_test = preprocess.shuffle_and_split_data(features, target, 0.2, 42)
        print(X_train)
        x_train_s = preprocess.numeric_scale_test_train_data(X_train, numeric_features_columns, categorical_features_columns[:-1])
        x_test_s = preprocess.numeric_scale_test_train_data(X_test, numeric_features_columns, categorical_features_columns[:-1])
        # y_train_s = preprocess.numeric_scale_test_train_data(y_train)
        # y_test_s = preprocess.numeric_scale_test_train_data(y_test)
        print(x_train_s)
        # Just for testing
        print('Features:', features)
        print('Target:', target)
        return x_train_s, x_test_s, y_train, y_test        
    
    else:
        print_data_error(error)
        return None, error
