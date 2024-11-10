import seaborn as sns # an abbrivation from a drama so as a meme NICE
import matplotlib.pyplot as plt

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
    plt.show()


def print_pairplot(numeric_data):
    print("Pairplot of numeric features:")

    # to adjust the size of the fonts
    with sns.plotting_context("notebook", rc={"axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5}):
        pairplot = sns.pairplot(numeric_data, diag_kind="hist", height=2.5, aspect=1.2)  # Adjust size and aspect ratio
        pairplot.fig.suptitle('Pairplot of Numeric Features', y=1.09)  
        plt.subplots_adjust(hspace=1, wspace=1)  # Adjust space between subplots
        plt.show()