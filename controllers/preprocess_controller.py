from helpers.load_data import load_data
import models.preprocess_model as preprocess
from views.display_view import print_data_error, print_data_success


def handle_preprocess():
    file_path = 'data/co2_emissions_data.csv'
    
    data, error = load_data(file_path)
    
    if data is not None:
        # So we will use the functions according to the target and features
        # depending on classification or regression I think
        features, target = preprocess.separate_features_and_target(data, 'Emission Class')
        features, target = preprocess.encode_categorical_features_targets(features, target)
        
        # Just for testing
        print('Features:', features)
        print('Target:', target)
        
    else:
        print_data_error(error)
