from controllers.perform_analysis_controller import handle_perform_analysis
from controllers.preprocess_controller import handle_preprocess


# So this is just an entry point to our app.
# Here I only call my processing controller.
# But later on we will update it to hanlde the whole flow of complete app.

# main function cause we will have more than on controller

    
def main():
    # handle_perform_analysis()
    handle_preprocess()


if __name__ == "__main__":
    main()