import models.preprocess_model as preprocess
import views.display_view as display

def handle_preprocessing():
    file_path = 'data/co2_emissions_data.csv'

    data, error = preprocess.load_data(file_path)

    if data is not None:
        display.print_data_success(data.shape[0], data.shape[1])

        missing_values = preprocess.check_missing_values(data)
        summary_stats = preprocess.check_numeric_features_scale(data)
        data_for_pairplot = preprocess.get_data_for_pairplot(data)
        correlation_data = preprocess.get_numeric_features_for_correlation(data)


        display.print_missing_values(missing_values)
        display.print_summary_stats(summary_stats)  
        display.print_pairplot(data_for_pairplot)
        display.print_correlation_heatmap(correlation_data)

    else:
        display.print_data_error(error)

