import os
import re
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

MIN_READS=1000
SAMPLES_LEGEND= "/cs/usr/rommy.amitai/Desktop/project/3rd_project/data/samples_legend.csv"

def get_file_name(file_path):
    return os.path.basename(file_path)

def parse_and_create_pd(file_path):
    file_name= get_file_name(file_path)
    pattern = r'\.(.*?)\.'
    matches = re.findall(pattern, file_name)
    if matches:
        sample_id = matches[0]
    else:
        sample_id="NONE FOUND"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    sequences = []
    counts = []

    for line in lines[2:]:  # Starting from the third line
        parts = line.strip().split('\t')
        sequence = ''.join(parts[6:])  # Extracting the sequence part

        # Check for '-' in the sequence and ensure count is a valid number
        if '-' not in sequence and parts[0].isdigit():
            sequences.append(sequence)
            counts.append(int(parts[0]))  # The count of the sequence

    # Creating a DataFrame
    df = pd.DataFrame({'Sequence': sequences, 'Count': counts})
    total_count = df['Count'].sum()
    #sequence_counts = df.groupby('Sequence')['Count'].sum()

    # Calculating percentages
    df['Percentages'] = (df['Count']/ total_count) * 100
    del df['Count']
    return df, total_count, sample_id

def pd_to_single_line(percentages, df, arg, sample_id, num_of_reads):
    result_dict = percentages.set_index('Sequence')['Percentages'].to_dict()
    seq_length= len(next(iter(result_dict)))

    #compute all known sequences, where doesn't exist add 0
    combos= generate_combos(seq_length)
    for combo in combos:
        if combo not in result_dict:
            result_dict[combo] = 0
    df = df.append([result_dict], ignore_index=False)
    df.loc[df.index[-1], 'Tag'] = arg
    df.loc[df.index[-1], 'num_of_reads'] = num_of_reads
    df.index = df.index[:-1].tolist() + [sample_id]
    #print(df)
    return df              

#return all combos of C,T as certain length
def generate_combos(length):
    def recursive_combos(prefix, remaining_length):
        if remaining_length == 0:
            combos.append(prefix)
            return
        for char in ['C', 'T']:
            recursive_combos(prefix + char, remaining_length - 1)
    if length <= 0:
        return []
    combos = []
    recursive_combos('', length)
    return combos

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def parse_all_files_folder(df, paths, arg):
    for file_path in paths:
        percentages, num_of_reads, sample_id = parse_and_create_pd(file_path)
        if (num_of_reads>MIN_READS): #ignore files with little reads
            df= pd_to_single_line(percentages, df, arg, sample_id, num_of_reads)
    return df

def get_staticstic_df(df):
    healthy_df = df[df['Tag'] == 0]
    # Calculate statistics (median, mean, std) for each pattern column in the healthy group
    pattern_columns = df.columns[1:]
    #pattern_columns= keep_only_panc(pattern_columns)
    healthy_statistics = healthy_df[pattern_columns].agg(['median', 'mean', 'std'])

    # Create new columns for the relative differences in statistics and z-scores
    for col in pattern_columns:
        median_diff_col = []
        mean_diff_col = []
        z_score_col = []
        z_score_median_col = []

        for index, row in df.iterrows():
            # Calculate the relative differences for each row in the current column
            median_diff = (row[col]) / (healthy_statistics.loc['median', col] + 1e-10)
            mean_diff = (row[col]) / (healthy_statistics.loc['mean', col] + 1e-10)
            z_score = (row[col] - healthy_statistics.loc['mean', col]) / (healthy_statistics.loc['std', col] + 1e-10)
            z_score_median = (row[col] - healthy_statistics.loc['median', col]) / (healthy_statistics.loc['std', col] + 1e-10)

            median_diff_col.append(median_diff)
            mean_diff_col.append(mean_diff)
            z_score_col.append(z_score)
            z_score_median_col.append(z_score_median)

        #df[f'{col}_Median_Relative_Diff'] = median_diff_col
        #df[f'{col}_Mean_Relative_Diff'] = mean_diff_col
        #df[f'{col}_Std_Relative_Diff'] = std_diff_col
        #df[f'{col}_Z-Score'] = z_score_col
        df[f'{col}_Z-Score-Median'] = z_score_median_col

    return df.drop(pattern_columns, axis=1) #remove all the raw data columns


def create_df_for_classification(yes_path, no_path):
    df = pd.DataFrame()
    df.insert(0, "Tag", 0)  # 1 for pdac, 0 for control
    paths = get_file_paths(yes_path)
    df = parse_all_files_folder(df, paths, 1)
    paths =get_file_paths(no_path)
    df = parse_all_files_folder(df, paths, 0)

    df_filtered = df.sort_values(by='num_of_reads', ascending=False).groupby(level=0).head(1).sort_index()
    df_filtered.drop(columns=['num_of_reads'], inplace=True)

    df_filtered = add_known_features(df_filtered)
    df_filtered = df_filtered[(df_filtered['stage'] == -1) | (df_filtered['stage'] >=1)]
    df_filtered.drop(columns=['stage'], inplace=True)
    #df_filtered = scale_df(df_filtered)
    return df_filtered

def classifiers_df_to_statistical(df):
    copy_df= df.copy()
    pdac_count = 0
    control_count = 0

    # Define a function to rename the 'Tag' values
    def rename_tag(x):
        nonlocal pdac_count, control_count  # Use nonlocal keyword to modify outer scope variables
        if x == 1:
            pdac_count += 1
            return f'PDAC{pdac_count}'
        else:
            control_count += 1
            return f'CONTROL{control_count}'

    #Apply the function to the 'Tag' column
    copy_df['Tag'] = copy_df['Tag'].apply(rename_tag)

    # Rename the columns based on the 'Tag' values
    transposed_df = copy_df.transpose()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df.columns.name = None
    transposed_df = transposed_df[1:]
    return transposed_df

def generate_label_ct_cnt(column):
    c_count = column.count('C')
    t_count = column.count('T')
    return f"{c_count}C{t_count}T"

def single_pat_to_ct_cnt(df):
    # Apply the function to create a list of labels
    labels = df.columns[1:].map(generate_label_ct_cnt)
    # Create an empty DataFrame to store the sums
    sums_df = pd.DataFrame(columns=labels, index=df.index)

    for index, row in df.iterrows():
        label_sums = {}
        # Iterate over each column (excluding the 'Tag' column)
        for col in df.columns[1:]:
            # Calculate the label for the current column
            label = generate_label_ct_cnt(col)
            # Add the value to the corresponding label sum
            label_sums[label] = label_sums.get(label, 0) + row[col]
        # Append the sums for the current row to the DataFrame
        #sums_df = sums_df.append(label_sums, ignore_index=True)
        sums_df.loc[index] = label_sums

    # Remove duplicate columns
    sums_df = sums_df.groupby(level=0, axis=1).first()
    # Concatenate the 'Tag' column from the original DataFrame
    sums_df.insert(0, 'Tag', df['Tag'].values)

    # Display the resulting DataFrame
    return sums_df

def generate_label_c_loc(column):
    # Find the positions of 'C' characters in the column name
    c_positions = [i for i, char in enumerate(column) if char == 'C']
    # Generate new column names based on the positions of 'C' characters
    fit_cols = [f'L{i + 1}MetPerc' for i in c_positions]
    # if not fit_cols:
    #     fit_cols = [f'L0Met']
    return fit_cols

def single_pat_to_c_loc_perc(df):
    # Apply the function to create a list of labels
    new_columns = [f'L{i}MetPerc' for i in range(1, len(df.columns[1]) + 1)]
    # Create an empty DataFrame to store the sums
    sums_df = pd.DataFrame(columns=new_columns, index=df.index)
    for index, row in df.iterrows():
        label_sums = {}
        # Iterate over each column (excluding the 'Tag' column)
        for col in df.columns[1:]:
            # Calculate the label for the current column
            labels = generate_label_c_loc(col)
            for label in labels:
                # Add the value to the corresponding label sum
                label_sums[label] = label_sums.get(label, 0) + row[col]
        # Append the sums for the current row to the DataFrame
        #sums_df = sums_df.append(label_sums, ignore_index=True)
        sums_df.loc[index] = label_sums
    # Concatenate the 'Tag' column from the original DataFrame
    sums_df.insert(0, 'Tag', df['Tag'].values)
    # Display the resulting DataFrame
    return sums_df

def complete_data_to_df(yes_path, no_path):
    pattern_df = create_df_for_classification(yes_path, no_path)
    ct_cnt_df = single_pat_to_ct_cnt(pattern_df)
    c_loc_df = single_pat_to_c_loc_perc(pattern_df)

    cnt_df_sel = ct_cnt_df.iloc[:, 1:]
    c_loc_df_sel = c_loc_df.iloc[:, 1:]
    main_df = pd.concat([pattern_df,
                         cnt_df_sel,
                         c_loc_df_sel], axis=1)
    return main_df

def complete_data_to_signi_pat(yes_path, no_path, test_func):
    pattern_df = create_df_for_classification(yes_path, no_path)
    ct_cnt_df = single_pat_to_ct_cnt(pattern_df)
    c_loc_df = single_pat_to_c_loc_perc(pattern_df)

    cnt_df_sel = ct_cnt_df.iloc[:, 1:]
    c_loc_df_sel = c_loc_df.iloc[:, 1:]
    main_df = pd.concat([pattern_df,
                         cnt_df_sel,
                         c_loc_df_sel], axis=1)
    print(main_df)
    #only take signi patterns to pca
    signi_patterns_all = test_func(classifiers_df_to_statistical(main_df))
    significant_patterns = np.append(signi_patterns_all['Pattern'].unique(), 'Tag')
    main_df = main_df.loc[:, main_df.columns.isin(significant_patterns)]

    return main_df

#might not need
def generate_patterns(sum_count):
    patterns = []
    for count_c in range(sum_count + 1):
        count_t = sum_count - count_c
        pattern = f"{count_c}C{count_t}T"
        patterns.append(pattern)
    return patterns

def add_known_features(df):
    pattern = re.compile(r'\d+')
    cols_to_keep = ['Sample ID', 'AJCC Stage']
    #value_mapping = {'T0':0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4}
    features= pd.read_csv(SAMPLES_LEGEND, header=1, usecols=cols_to_keep)
    features.rename(columns={'Sample ID': 'sample_id', 'AJCC Stage': 'stage'}, inplace=True)
    #features['stage'].fillna(-1, inplace=True)
    #features['stage'].replace(value_mapping, inplace=True)
    features['stage'] = features['stage'].apply(lambda x: int(pattern.findall(str(x))[0]) if pd.notna(x) else -1)
    merged_index_df = pd.merge(df, features, left_index=True, right_on='sample_id', how='left')
    merged_index_df.set_index('sample_id', inplace=True)
    return merged_index_df

def scale_df(df):
    scaler = MinMaxScaler()
    #numeric_cols = df.select_dtypes(include=['number']).columns.drop('Tag')
    numeric_cols = df.columns.drop('Tag')
    # Fit and transform the scaler on the selected features
    df_scaled = df.copy()  # Create a copy of the DataFrame to retain the original data
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # Now df_scaled contains the scaled values while preserving the DataFrame structure
    return df_scaled

def process_folders(base_directory):
    # Iterate through folders in the base directory
    for folder_name in os.listdir(base_directory):
        # Construct paths for "yes" and "no" folders within each folder
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        # Check if both "yes" and "no" folders exist
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            # Process the data in the current pair of "yes" and "no" folders
            if folder_name=='auto-1':
                main_df =create_df_for_classification(yes_folder, no_folder)
                main_df= add_known_features(main_df)
                #main_df= scale_df(main_df)
                print(main_df)

def main():
    base_directory = "/cs/usr/rommy.amitai/Desktop/project/plasma_yes_no_marker"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
