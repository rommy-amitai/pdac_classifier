import os
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

MIN_READS = 1000
SAMPLES_LEGEND = "/samples_legend.csv"

def get_file_name(file_path):
    """
    Extracts the file name from the file path.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    str: The file name.
    """
    return os.path.basename(file_path)

def parse_and_create_pd(file_path):
    """
    Parses the file and creates a DataFrame.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    DataFrame, int, str: The parsed DataFrame, total count of reads, and sample ID.
    """
    file_name = get_file_name(file_path)
    pattern = r'\.(.*?)\.'
    matches = re.findall(pattern, file_name)
    sample_id = matches[0] if matches else "NONE FOUND"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    sequences, counts = [], []
    for line in lines[2:]:  # Starting from the third line
        parts = line.strip().split('\t')
        sequence = ''.join(parts[6:])  # Extracting the sequence part
        if '-' not in sequence and parts[0].isdigit():
            sequences.append(sequence)
            counts.append(int(parts[0]))  # The count of the sequence

    df = pd.DataFrame({'Sequence': sequences, 'Percentages': (np.array(counts) / sum(counts)) * 100})
    return df, sum(counts), sample_id

def pd_to_single_line(percentages, df, arg, sample_id, num_of_reads):
    """
    Converts percentages DataFrame to a single line DataFrame.

    Parameters:
    percentages (DataFrame): DataFrame with sequence percentages.
    df (DataFrame): The main DataFrame to append to.
    arg (int): The tag value (1 for PDAC, 0 for control).
    sample_id (str): The sample ID.
    num_of_reads (int): The number of reads.

    Returns:
    DataFrame: The updated main DataFrame.
    """
    result_dict = percentages.set_index('Sequence')['Percentages'].to_dict()
    seq_length = len(next(iter(result_dict)))
    combos = generate_combos(seq_length)
    for combo in combos:
        if combo not in result_dict:
            result_dict[combo] = 0
    df = df.append([result_dict], ignore_index=False)
    df.loc[df.index[-1], 'Tag'] = arg
    df.loc[df.index[-1], 'num_of_reads'] = num_of_reads
    df.index = df.index[:-1].tolist() + [sample_id]
    return df

def generate_combos(length):
    """
    Generates all possible combinations of 'C' and 'T' of a given length.

    Parameters:
    length (int): The length of the combinations.

    Returns:
    list: List of all combinations.
    """
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
    """
    Retrieves all file paths within a folder.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    list: List of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def parse_all_files_folder(df, paths, arg):
    """
    Parses all files in a folder and appends data to the DataFrame.

    Parameters:
    df (DataFrame): The main DataFrame.
    paths (list): List of file paths to parse.
    arg (int): The tag value (1 for PDAC, 0 for control).

    Returns:
    DataFrame: The updated main DataFrame.
    """
    for file_path in paths:
        percentages, num_of_reads, sample_id = parse_and_create_pd(file_path)
        if num_of_reads > MIN_READS:
            df = pd_to_single_line(percentages, df, arg, sample_id, num_of_reads)
    return df

def get_staticstic_df(df):
    """
    Calculates statistics and returns a DataFrame with z-scores.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with z-scores.
    """
    healthy_df = df[df['Tag'] == 0]
    pattern_columns = df.columns[1:]
    healthy_statistics = healthy_df[pattern_columns].agg(['median', 'mean', 'std'])

    for col in pattern_columns:
        z_score_median_col = [
            (row[col] - healthy_statistics.loc['median', col]) / (healthy_statistics.loc['std', col] + 1e-10)
            for _, row in df.iterrows()
        ]
        df[f'{col}_Z-Score-Median'] = z_score_median_col

    return df.drop(pattern_columns, axis=1)

def create_df_for_classification(yes_path, no_path):
    """
    Creates a DataFrame for classification from yes and no paths.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.

    Returns:
    DataFrame: The combined and filtered DataFrame.
    """
    df = pd.DataFrame(columns=['Tag'])
    df.insert(0, "Tag", 0)  # 1 for PDAC, 0 for control

    paths = get_file_paths(yes_path)
    df = parse_all_files_folder(df, paths, 1)
    paths = get_file_paths(no_path)
    df = parse_all_files_folder(df, paths, 0)

    df_filtered = df.sort_values(by='num_of_reads', ascending=False).groupby(level=0).head(1).sort_index()
    df_filtered.drop(columns=['num_of_reads'], inplace=True)

    df_filtered = add_known_features(df_filtered)
    df_filtered = df_filtered[(df_filtered['stage'] == -1) | (df_filtered['stage'] >= 1)]
    df_filtered.drop(columns=['stage'], inplace=True)
    return df_filtered

def classifiers_df_to_statistical(df):
    """
    Transposes and renames the DataFrame for statistical analysis.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The transposed and renamed DataFrame.
    """
    copy_df = df.copy()
    pdac_count, control_count = 0, 0

    def rename_tag(x):
        nonlocal pdac_count, control_count
        if x == 1:
            pdac_count += 1
            return f'PDAC{pdac_count}'
        else:
            control_count += 1
            return f'CONTROL{control_count}'

    copy_df['Tag'] = copy_df['Tag'].apply(rename_tag)
    transposed_df = copy_df.transpose()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df[1:]
    return transposed_df

def generate_label_ct_cnt(column):
    """
    Generates a label based on the count of 'C' and 'T' in the column name.

    Parameters:
    column (str): The column name.

    Returns:
    str: The generated label.
    """
    c_count = column.count('C')
    t_count = column.count('T')
    return f"{c_count}C{t_count}T"

def single_pat_to_ct_cnt(df):
    """
    Converts single pattern DataFrame to count DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The count DataFrame.
    """
    labels = df.columns[1:].map(generate_label_ct_cnt)
    sums_df = pd.DataFrame(columns=labels, index=df.index)
    for index, row in df.iterrows():
        label_sums = {}
        for col in df.columns[1:]:
            label = generate_label_ct_cnt(col)
            label_sums[label] = label_sums.get(label, 0) + row[col]
        sums_df.loc[index] = label_sums
    sums_df.insert(0, 'Tag', df['Tag'].values)
    return sums_df.groupby(level=0, axis=1).first()

def generate_label_c_loc(column):
    """
    Generates new column names based on the positions of 'C' characters.

    Parameters:
    column (str): The column name.

    Returns:
    list: List of generated labels.
    """
    c_positions = [i for i, char in enumerate(column) if char == 'C']
    return [f'L{i + 1}MetPerc' for i in c_positions]

def single_pat_to_c_loc_perc(df):
    """
    Converts single pattern DataFrame to location percentage DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The location percentage DataFrame.
    """
    new_columns = [f'L{i}MetPerc' for i in range(1, len(df.columns[1]) + 1)]
    sums_df = pd.DataFrame(columns=new_columns, index=df.index)
    for index, row in df.iterrows():
        label_sums = {}
        for col in df.columns[1:]:
            labels = generate_label_c_loc(col)
            for label in labels:
                label_sums[label] = label_sums.get(label, 0) + row[col]
        sums_df.loc[index] = label_sums
    sums_df.insert(0, 'Tag', df['Tag'].values)
    return sums_df

def complete_data_to_df(yes_path, no_path):
    """
    Completes the data processing to generate the final DataFrame.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.

    Returns:
    DataFrame: The combined DataFrame.
    """
    pattern_df = create_df_for_classification(yes_path, no_path)
    ct_cnt_df = single_pat_to_ct_cnt(pattern_df)
    c_loc_df = single_pat_to_c_loc_perc(pattern_df)

    main_df = pd.concat([pattern_df, ct_cnt_df.iloc[:, 1:], c_loc_df.iloc[:, 1:]], axis=1)
    return main_df

def complete_data_to_signi_pat(yes_path, no_path, test_func):
    """
    Completes the data processing to generate a DataFrame with significant patterns.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.
    test_func (function): Function to test for significant patterns.

    Returns:
    DataFrame: The DataFrame with significant patterns.
    """
    pattern_df = create_df_for_classification(yes_path, no_path)
    ct_cnt_df = single_pat_to_ct_cnt(pattern_df)
    c_loc_df = single_pat_to_c_loc_perc(pattern_df)

    main_df = pd.concat([pattern_df, ct_cnt_df.iloc[:, 1:], c_loc_df.iloc[:, 1:]], axis=1)
    signi_patterns_all = test_func(classifiers_df_to_statistical(main_df))
    significant_patterns = np.append(signi_patterns_all['Pattern'].unique(), 'Tag')
    main_df = main_df.loc[:, main_df.columns.isin(significant_patterns)]

    return main_df

def add_known_features(df):
    """
    Adds known features to the DataFrame from a CSV file.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The updated DataFrame with known features.
    """
    pattern = re.compile(r'\d+')
    features = pd.read_csv(SAMPLES_LEGEND, header=1, usecols=['Sample ID', 'AJCC Stage'])
    features.rename(columns={'Sample ID': 'sample_id', 'AJCC Stage': 'stage'}, inplace=True)
    features['stage'] = features['stage'].apply(lambda x: int(pattern.findall(str(x))[0]) if pd.notna(x) else -1)
    merged_index_df = pd.merge(df, features, left_index=True, right_on='sample_id', how='left')
    merged_index_df.set_index('sample_id', inplace=True)
    return merged_index_df

def scale_df(df):
    """
    Scales the DataFrame using Min-Max scaling.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The scaled DataFrame.
    """
    scaler = MinMaxScaler()
    numeric_cols = df.columns.drop('Tag')
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled

def process_folders(base_directory):
    """
    Processes each folder in the base directory.

    Parameters:
    base_directory (str): The base directory containing data folders.
    """
    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            if folder_name == 'auto-1':
                main_df = create_df_for_classification(yes_folder, no_folder)
                main_df = add_known_features(main_df)
                print(main_df)

def create_full_df(yes_path, no_path, folder_name, main_df):
    """
    Creates a full DataFrame from the yes and no paths.

    Parameters:
    yes_path (str): Path to the "yes" data.
    no_path (str): Path to the "no" data.
    folder_name (str): Current folder name.
    main_df (DataFrame): Main DataFrame to append to.

    Returns:
    DataFrame: Combined DataFrame.
    """
    pattern_df = create_df_for_classification(yes_path, no_path)
    ct_cnt_df = single_pat_to_ct_cnt(pattern_df)
    c_loc_df = single_pat_to_c_loc_perc(pattern_df)

    pattern_df.columns = [f'{col}_pattern' if 'Tag' not in col else col for col in pattern_df.columns]
    ct_cnt_df.columns = [f'{col}_ct_cnt' if 'Tag' not in col else col for col in ct_cnt_df.columns]
    c_loc_df.columns = [f'{col}_c_loc_perc' if 'Tag' not in col else col for col in c_loc_df.columns]

    pattern_df = scale_df(pattern_df)
    ct_cnt_df = scale_df(ct_cnt_df)
    c_loc_df = scale_df(c_loc_df)

    cnt_df_sel = ct_cnt_df.iloc[:, 1:]
    c_loc_df_sel = c_loc_df.iloc[:, 1:]
    local_df = pd.concat([pattern_df, cnt_df_sel, c_loc_df_sel], axis=1)

    new_column_names_local = [(f'{folder_name}_' + col) if col != 'Tag' else col for col in local_df.columns]
    local_df.columns = new_column_names_local

    if not main_df.empty:
        local_df = pd.merge(local_df, main_df, left_index=True, right_index=True, suffixes=('', '_drop'))
        local_df = local_df.loc[:, ~local_df.columns.str.endswith('_drop')]

    return local_df

def main():
    """
    Main function to process folders in the specified base directory.
    """
    base_directory = "/plasma_markers"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
