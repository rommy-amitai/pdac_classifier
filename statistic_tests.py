from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_parser as dp
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

MIN_SAMPLES=10
ALPHA=0.2

def perform_ks_test(df):
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]

    ks_test_results = []

    for pattern in df.index:
        # Extract counts for the current pattern from each group
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]

        # Perform KS test
        stat, p_value = ks_2samp(pdac_values, control_values)
        ks_test_results.append({'Pattern': pattern, 'statistic': stat, 'p_value': p_value})
        
    ks_test_df = pd.DataFrame(ks_test_results)
    ks_test_df['fdr'] = multipletests(ks_test_df['p_value'], method='fdr_bh')[1]  # Calculate FDR-adjusted p-values
    sorted_df = ks_test_df.sort_values(by='fdr')
    significant_df = sorted_df[sorted_df['fdr'] < ALPHA]
    return significant_df


def perform_t_test(df, thresh=None):
    # Filter the columns for 'PDAC' and 'CONTROL'
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]

    t_test_results = []

    # Perform t-test for each pattern
    for pattern in df.index:
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]

        #if (pdac_values > 0).sum() >= MIN_SAMPLES or (control_values > 0).sum() >= MIN_SAMPLES:

        # Perform independent t-test
        t_statistic, p_value = ttest_ind(pdac_values, control_values)
        # Store t-test result
        t_test_results.append({'Pattern': pattern, 't_statistic': t_statistic, 'p_value': p_value})

    # Convert results to DataFrame
    t_test_df = pd.DataFrame(t_test_results).dropna()
    if thresh:
        t_test_df['fdr'] = multipletests(t_test_df['p_value'], method='fdr_bh')[1]  # Calculate FDR-adjusted p-values
        sorted_df = t_test_df.sort_values(by='fdr')
        return sorted_df[sorted_df['fdr'] < ALPHA]
    else:
        return t_test_df.sort_values(by='p_value')

def pd_to_bar(df, folder_name):
    # Filter the columns for 'PDAC' and 'CONTROL'
    pdac_columns = df.columns[df.columns.str.contains('PDAC')]
    control_columns = df.columns[df.columns.str.contains('CONTROL')]
    pdac_df = df[pdac_columns]
    control_df = df[control_columns]
    # Plot bars for each pattern
    for pattern in df.index:
        # Get PDAC and CONTROL values for the current pattern
        pdac_values = pdac_df.loc[pattern]
        control_values = control_df.loc[pattern]

        #make sure at least one group has MIN_SAMPLES or more samples which aren't empty
        if (pdac_values > 0).sum() >= MIN_SAMPLES or (control_values > 0).sum() >= MIN_SAMPLES:
            fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size

            # Sort PDAC and CONTROL values
            pdac_sorted_indices = np.argsort(pdac_values)
            control_sorted_indices = np.argsort(control_values)[::-1]

            # Calculate the positions for PDAC and CONTROL bars
            pdac_positions = np.arange(len(pdac_values))
            control_positions = np.arange(len(control_values)) + len(pdac_values) + 1

            # Plot PDAC bars
            ax.bar(pdac_positions, pdac_values[pdac_sorted_indices], width=0.4, color='blue',
                   label='PDAC')  # Adjust width

            # Plot CONTROL bars on the other side of y-axis
            ax.bar(control_positions, control_values[control_sorted_indices], width=0.4, color='red',
                   label='CONTROL')  # Adjust width

            # Remove x-axis tick labels
            ax.set_xticks([])

            # Set y-axis label
            ax.set_ylabel('Percentage')

            # Add labels and legend
            ax.set_title(f'{folder_name} Pattern Distribution - {pattern}')
            ax.legend()

            # Save the plot to the figs folder
            if not os.path.exists(f'figs/{folder_name}'):
                os.makedirs(f'figs/{folder_name}')
            plt.savefig(f'figs/{folder_name}/{pattern}_distribution.png',
                        bbox_inches='tight')  # Use bbox_inches='tight' to avoid cutting off labels

            # Close the plot to release memory
            plt.close()

def process_folders(base_directory):
    # Iterate through folders in the base directory
    for folder_name in os.listdir(base_directory):
        # Construct paths for "yes" and "no" folders within each folder
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        # Check if both "yes" and "no" folders exist
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            # Process the data in the current pair of "yes" and "no" folders
            start_process_single_patterns(yes_folder, no_folder, folder_name)
            #start_process_ct_counter(yes_folder, no_folder, folder_name)
            #start_process_c_loc(yes_folder, no_folder, folder_name)

def start_process_single_patterns(yes_path, no_path, folder_name):
    df = dp.create_df_for_classification(yes_path, no_path)
    df_static= dp.classifiers_df_to_statistical(df)
    #pd_to_bar(df_static, folder_name)
    run_tests(df_static, folder_name)


def start_process_ct_counter(yes_path, no_path, folder_name):
    df = dp.single_pat_to_ct_cnt(dp.create_df_for_classification(yes_path, no_path))
    df_static = dp.classifiers_df_to_statistical(df)
    #pd_to_bar(df_static, folder_name)
    run_tests(df_static, folder_name)


def start_process_c_loc(yes_path, no_path, folder_name):
    df = dp.single_pat_to_c_loc_perc(dp.create_df_for_classification(yes_path, no_path))
    df_static = dp.classifiers_df_to_statistical(df)
    #pd_to_bar(df_static, folder_name)
    run_tests(df_static, folder_name)


def run_tests(df, folder_name):
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
    base_directory = "/cs/usr/rommy.amitai/Desktop/project/plasma_yes_no_marker"
    process_folders(base_directory)

if __name__ == "__main__":
    main()