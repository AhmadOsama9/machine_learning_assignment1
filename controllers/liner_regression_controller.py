from models.linear_regression_model import select_features, model, eval_model_using_r2_score
from views.display_view import print_cost_pairplot

def handle_model(x_train, x_test, y_train, y_test):
    x_train = select_features(x_train)
    x_test = select_features(x_test)

    learning_rate = 0.009
    iterarions = 2000

    # hist, w, b = model(x_train, y_train, learning_rate, iterarions)
    model(x_train, y_train, learning_rate, iterarions)

    # print_cost_pairplot(hist)
    # print("final cost :", hist[-1])


    # r2_score = eval_model_using_r2_score((w, b), x_test, y_test)

    # print("R2 Score:", r2_score)