from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
# the 2 features we will choose is 
# feature 1 : fuel consumption comb (mpg) with feature 2 : engine size 
# as they both are highly correaleted with the target and is not highly correaleted with each other


def select_features(data):
    selected_data = data[['Fuel Consumption Comb (mpg)', 'Engine Size(L)']]
    # selected_data = data[['Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']]
    # selected_data = data[['Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']]
    # selected_data = data[['Fuel Consumption Comb (L/100 km)', 'Cylinders']]

    return selected_data

# def model(features, target, learning_rate, iterations, tolerance=1e-6):
#     model = LinearRegression()
#     model.fit(features, target)
#     predictions = model.predict(features)
#     r2 = r2_score(target, predictions)
#     final_cost = mean_squared_error(target, predictions) / 2

#     print( {
#         "R2 Score": r2,
#         "Final Cost": final_cost,
#         "Model": model,
#         "Predictions": predictions
#     })

def model(features, target, learning_rate, iterations, tolerance=1e-6):


    if isinstance(features, pd.DataFrame):
        features = features.values

    if isinstance(target, pd.DataFrame):
        target = target.values
    
    W = np.random.uniform(-5, 5, size=(features.shape[1], 1))  # Small random values
    b = np.zeros(1)
    cost_history = []

    prev_cost = float('inf')
    
    # Training Loop
    for i in range(iterations):
        # Step 1: Predict
        y_pred = np.dot(features, W) + b
        
        # Step 2: Compute mse
        m = len(target)
        error = y_pred - target.reshape(-1, 1)
        squared_error = error ** 2
        mse = sum(squared_error) / (2 * m)
        # print("mse", mse)
        cost_history.append(mse)
        
         # Check for convergence: if the cost doesn't decrease significantly
        if abs(prev_cost - mse) < tolerance:
            print("Convergence reached at iteration", i)
            break
        
        # Update previous cost
        prev_cost = mse


        # Step 3: Compute derivatives
        dW = (1 / m) * np.dot(features.T, error)
        db = (1 / m) * np.sum(error) 
        
        # Step 4: Update parameters
        W -= learning_rate * dW
        b -= learning_rate * db


    return cost_history, W, b


def eval_model_using_r2_score(wieghts, features, target):
    y_pred = np.dot(features, wieghts[0]) + wieghts[1]
    r2 = r2_score(target, y_pred)
    return r2
