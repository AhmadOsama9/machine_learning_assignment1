import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data, None
    except Exception as e:  
        return None, str(e)




def select_features(data):
    selected_data = data[['Fuel Consumption Comb (L/100 km)', 'Engine Size(L)']]
    return selected_data

def model(features, target, learning_rate, iterations, tolerance=1e-6):
    if isinstance(features, pd.DataFrame):
        features = features.values

    if isinstance(target, pd.DataFrame):
        target = target.values
    
    W = np.random.uniform(-1.0, 1.0, size=(features.shape[1], 1))  # Small random values
    b = np.zeros(1)
    cost_history = []

    prev_cost = float('inf')
    
    # Training Loop
    for i in range(iterations):
        # Step 1: Predict
        y_pred = np.dot(features, W) + b
        
        # Step 2: Compute mse
        m = len(target)
        error = y_pred - target.to_numpy().reshape(-1, 1)
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

def check_missing_values(data):
    # will data.isnull return a series where index is column and value is the count of missing
    missing_values = data.isnull().sum()
    return missing_values # could use missine_values[missing_values > 0] to filter out columns with no missing values


def check_numeric_features_scale(data):
    numeric_features = data.select_dtypes(include = ['float64', 'int64'])
    return numeric_features.describe()

def get_data_for_pairplot(data):
    return data.select_dtypes(include = ['float64', 'int64']) 
    # In the assignment it didn't specify pairplot for only numeric so I'm confused

def get_numeric_features_for_correlation(data):
    return data.select_dtypes(include = ['float64', 'int64'])



def separate_features_and_target(data, target_column):
    features = data.drop(columns=[target_column]) # we drop the target which will be CO2 emission in regression and Emission class in classification
    target = data[target_column]
    return features, target 
    
# label encoder
def encode_categorical_features_targets_using_LE(features, target):

    le = LabelEncoder()

    categorical_columns = features.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        features[col] = le.fit_transform(features[col])

    if isinstance(target, pd.DataFrame):
        target_columns = target.select_dtypes(include=['object']).columns
        for col in target_columns:
            target[col] = le.fit_transform(target[col])
    elif target.dtype == 'object':
        target = le.fit_transform(target)
    
    return features, target



def shuffle_and_split_data(features, target1, target2, test_size=0.2, random=42):
    X_train, X_test, y_train1, y_test1, y_train2, y_test2 =  train_test_split(features, target1, target2, test_size=test_size, random_state=random)
    return X_train, X_test, y_train1, y_test1, y_train2, y_test2


def numeric_scale_test_train_data(data, numeric_features_columns, categorical_features_columns):
    scaler = StandardScaler()

    data_n = data[numeric_features_columns]

    scaled_data = scaler.fit_transform(data_n)
    
    scaled_data_df = pd.DataFrame(scaled_data, columns=numeric_features_columns, index=data.index)
   
    
    categorical_data = data[categorical_features_columns]

    final_data = pd.concat([scaled_data_df, categorical_data], axis=1)
    return final_data   


def print_data_error(error):
    print(f"Error loading dataset: {error}")

def print_data_success(rows, columns):
    print(f"Dataset loaded successfully rows: {rows} and columns: {columns}")


def print_missing_values(missing_values):
    print("Missing values in dataset:")
    print(missing_values)


def print_summary_stats(summary_stats):
    print("Summary statistics of numeric features:")
    print(summary_stats)


def print_correlation_heatmap(correlation_data):
    print("Correlation heatmap of numeric features:")
    correlation_matrix = correlation_data.corr()
    print(correlation_matrix)
    
    plt.figure(figsize=(12, 8))  # To control the size of the heatmap
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()  # To control the padding
    # plt.show()


def print_pairplot(numeric_data):
    print("Pairplot of numeric features:")

    # to adjust the size of the fonts
    with sns.plotting_context("notebook", rc={"axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5}):
        pairplot = sns.pairplot(numeric_data, diag_kind="hist", height=2.5, aspect=1.2)  # Adjust size and aspect ratio
        pairplot.fig.suptitle('Pairplot of Numeric Features', y=1.09)  
        plt.subplots_adjust(hspace=1, wspace=1)  # Adjust space between subplots
        # plt.show()
        plt.savefig('pairplot.png')



def print_cost_pairplot(costs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(costs) + 1), costs, label='Cost', color='blue', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations in Gradient Descent')
    plt.grid(True)
    plt.legend()
    plt.show()



def handle_linear_model(x_train, x_test, y_train, y_test):
    x_train = select_features(x_train)
    x_test = select_features(x_test)

    learning_rate = 0.01
    iterarions = 1000

    hist, w, b = model(x_train, y_train, learning_rate, iterarions)
    # print_cost_pairplot(hist)

    print("final cost :", hist[-1])
    r2_score = eval_model_using_r2_score((w, b), x_test, y_test)
    print("R2 Score:", r2_score)


def handle_logisitic_model(x_train, x_test, y_train, y_test):
    x_train = select_features(x_train)
    x_test = select_features(x_test)

    model = SGDClassifier()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Logistic Regression Accuracy:", accuracy) 
    


def handle_perform_analysis():
    file_path = 'data/co2_emissions_data.csv'

    data, error = load_data(file_path)

    if data is not None:
        display.print_data_success(data.shape[0], data.shape[1])

        missing_values = perform_analysis.check_missing_values(data)
        summary_stats = perform_analysis.check_numeric_features_scale(data)
        data_for_pairplot = perform_analysis.get_data_for_pairplot(data)
        correlation_data = perform_analysis.get_numeric_features_for_correlation(data)


        display.print_missing_values(missing_values)
        display.print_summary_stats(summary_stats)  
        display.print_pairplot(data_for_pairplot)
        display.print_correlation_heatmap(correlation_data)

    else:
        display.print_data_error(error)


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


