import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.stats.multitest import multipletests
import data_parser as dp

MIN_SAMPLES = 10
ALPHA = 0.2

def perform_ks_test(df):
    """
    Performs the Kolmogorov-Smirnov test on the given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: A DataFrame containing significant patterns based on the KS test.
    """
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]

    ks_test_results = []
    for pattern in df.index:
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]
        stat, p_value = ks_2samp(pdac_values, control_values)
        ks_test_results.append({'Pattern': pattern, 'statistic': stat, 'p_value': p_value})
        
    ks_test_df = pd.DataFrame(ks_test_results)
    ks_test_df['fdr'] = multipletests(ks_test_df['p_value'], method='fdr_bh')[1]  # FDR-adjusted p-values
    significant_df = ks_test_df.sort_values(by='fdr').query('fdr < @ALPHA')
    return significant_df

def perform_t_test(df, thresh=None):
    """
    Performs the t-test on the given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    thresh (float): Threshold for FDR adjustment.

    Returns:
    DataFrame: A DataFrame containing significant patterns based on the t-test.
    """
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]

    t_test_results = []
    for pattern in df.index:
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]
        t_statistic, p_value = ttest_ind(pdac_values, control_values)
        t_test_results.append({'Pattern': pattern, 't_statistic': t_statistic, 'p_value': p_value})

    t_test_df = pd.DataFrame(t_test_results).dropna()
    if thresh:
        t_test_df['fdr'] = multipletests(t_test_df['p_value'], method='fdr_bh')[1]  # FDR-adjusted p-values
        return t_test_df.sort_values(by='fdr').query('fdr < @ALPHA')
    return t_test_df.sort_values(by='p_value')

def pd_to_bar(df, folder_name):
    """
    Generates bar plots for PDAC and CONTROL distributions.

    Parameters:
    df (DataFrame): The input DataFrame.
    folder_name (str): The name of the folder to save the plots.
    """
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]

    for pattern in df.index:
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]

        if (pdac_values > 0).sum() >= MIN_SAMPLES or (control_values > 0).sum() >= MIN_SAMPLES:
            fig, ax = plt.subplots(figsize=(10, 6))

            pdac_sorted_indices = np.argsort(pdac_values)
            control_sorted_indices = np.argsort(control_values)[::-1]

            pdac_positions = np.arange(len(pdac_values))
            control_positions = np.arange(len(control_values)) + len(pdac_values) + 1

            ax.bar(pdac_positions, pdac_values[pdac_sorted_indices], width=0.4, color='blue', label='PDAC')
            ax.bar(control_positions, control_values[control_sorted_indices], width=0.4, color='red', label='CONTROL')

            ax.set_xticks([])
            ax.set_ylabel('Percentage')
            ax.set_title(f'{folder_name} Pattern Distribution - {pattern}')
            ax.legend()

            if not os.path.exists(f'figs/{folder_name}'):
                os.makedirs(f'figs/{folder_name}')
            plt.savefig(f'figs/{folder_name}/{pattern}_distribution.png', bbox_inches='tight')
            plt.close()

def process_folders(base_directory):
    """
    Processes each folder in the base directory and performs tests.

    Parameters:
    base_directory (str): The base directory containing data folders.
    """
    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            start_process_single_patterns(yes_folder, no_folder, folder_name)

def start_process_single_patterns(yes_path, no_path, folder_name):
    """
    Starts processing single patterns and runs tests.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.
    folder_name (str): The name of the folder being processed.
    """
    df = dp.create_df_for_classification(yes_path, no_path)
    df_static = dp.classifiers_df_to_statistical(df)
    run_tests(df_static, folder_name)

def start_process_ct_counter(yes_path, no_path, folder_name):
    """
    Starts processing CT counters and runs tests.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.
    folder_name (str): The name of the folder being processed.
    """
    df = dp.single_pat_to_ct_cnt(dp.create_df_for_classification(yes_path, no_path))
    df_static = dp.classifiers_df_to_statistical(df)
    run_tests(df_static, folder_name)

def start_process_c_loc(yes_path, no_path, folder_name):
    """
    Starts processing C locations and runs tests.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.
    folder_name (str): The name of the folder being processed.
    """
    df = dp.single_pat_to_c_loc_perc(dp.create_df_for_classification(yes_path, no_path))
    df_static = dp.classifiers_df_to_statistical(df)
    run_tests(df_static, folder_name)

def run_tests(df, folder_name):
    """
    Runs KS and t-tests on the DataFrame and prints results.

    Parameters:
    df (DataFrame): The input DataFrame.
    folder_name (str): The name of the folder being processed.
    """
    print(folder_name)
    res = perform_ks_test(df)
    if not res.empty:
        print('Kolmogorov-Smirnov test:')
        print(res)
        print()
    res = perform_t_test(df)
    if not res.empty:
        print('T test with FDR correction:')
        print(res)
        print()

def main():
    """
    Main function to process folders in the specified base directory.
    """
    base_directory = "/plasma_markers"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
