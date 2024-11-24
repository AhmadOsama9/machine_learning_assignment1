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
        print('CO2 Emissions(g/km)' in numeric_features_columns)
        # So we will use the functions according to the target and features
        # depending on classification or regression I think
        features, target1 = preprocess.separate_features_and_target(data, 'CO2 Emissions(g/km)')
        features, target2 = preprocess.separate_features_and_target(data, 'Emission Class')
        features, target2 = preprocess.encode_categorical_features_targets_using_LE(features, target2)
        X_train, X_test, y_train1, y_test1, y_train2, y_test2 = preprocess.shuffle_and_split_data(features, target1, target2, 0.2, 42)
   
        x_train_s = preprocess.numeric_scale_test_train_data(X_train, numeric_features_columns[:-1], categorical_features_columns[:-1])
        x_test_s = preprocess.numeric_scale_test_train_data(X_test, numeric_features_columns[:-1], categorical_features_columns[:-1])
        # Just for testing
        print('Features:', features)
        print('Target1:', target1)
        print()
        print('Target2:', target2)
        return x_train_s, x_test_s, y_train1, y_test1, y_train2, y_test2        
    
    else:
        print_data_error(error)
        return None, error
