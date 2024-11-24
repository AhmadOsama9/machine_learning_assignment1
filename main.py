from controllers.perform_analysis_controller import handle_perform_analysis
from controllers.preprocess_controller import handle_preprocess
from controllers.liner_regression_controller import handle_model

# So this is just an entry point to our app.
# Here I only call my processing controller.
# But later on we will update it to hanlde the whole flow of complete app.

# main function cause we will have more than on controller

    
def main():
    # handle_perform_analysis()
    x_train_s, x_test_s, y_train, y_test = handle_preprocess()
    handle_model(x_train_s, x_test_s, y_train, y_test)

if __name__ == "__main__":
    main()