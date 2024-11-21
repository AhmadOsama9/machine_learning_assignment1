from helpers.load_data import load_data
import models.perform_analysis_model as perform_analysis
import views.display_view as display

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

