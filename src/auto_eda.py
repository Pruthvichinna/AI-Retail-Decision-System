from ydata_profiling import ProfileReport
from preprocess import load_and_preprocess_data

def generate_eda_report(data_path, output_path):
    """
    Loads and preprocesses data, then generates a comprehensive EDA report.

    Args:
        data_path (str): Path to the raw data directory.
        output_path (str): Path to save the output HTML report.
    """
    print("Loading and preprocessing data...")
    master_df = load_and_preprocess_data(data_path)
    
    print("Generating EDA profile report...")
    profile = ProfileReport(
        master_df,
        title="Retail Analytics EDA Report",
        explorative=True
    )
    
    print(f"Saving report to {output_path}...")
    profile.to_file(output_path)
    print("✅ EDA report generated successfully.")

if __name__ == '__main__':
    # Define the relative paths
    path_to_data = '../data'
    report_output_path = '../docs/eda_report.html'
    
    # Generate the report
    generate_eda_report(path_to_data, report_output_path)