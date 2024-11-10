from controllers.preprocess_controller import handle_preprocessing

# So this is just an entry point to our app.
# Here I only call my processing controller.
# But later on we will update it to hanlde the whole flow of complete app.

if __name__ == "__main__":
    handle_preprocessing()