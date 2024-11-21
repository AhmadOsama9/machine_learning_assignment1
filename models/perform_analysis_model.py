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

