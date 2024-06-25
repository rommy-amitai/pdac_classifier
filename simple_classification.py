import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_parser as dp
import statistic_tests as hp

RANDOM_STATE = 42

# Define the models to be used
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Support Vector Machine (poly)', SVC(kernel='poly', probability=True)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('XGBoost', xgb.XGBClassifier())  # Adding XGBoost
]

def split_df(df, test_size=0.2, random_state=RANDOM_STATE):
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    y = df.iloc[:, 0]  # Assuming the first column is the target
    X = df.iloc[:, 1:]  # Assuming the remaining columns are features
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_roc(model_name, y, y_pred_proba, desc, plot=True):
    """
    Creates and plots the ROC curve.

    Parameters:
    model_name (str): The name of the model.
    y (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities.
    desc (str): Description for the plot title.
    plot (bool): Whether to plot the ROC curve.

    Returns:
    float: Best threshold value.
    """
    auc = roc_auc_score(y, y_pred_proba)
    fpr, tpr, threshold = roc_curve(y, y_pred_proba)
    t = find_best_threshold(threshold, fpr, tpr)
    if plot:
        plt.plot(fpr, tpr, label="AUC=" + str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.title(f'{desc.upper()} | ROC Curve for {model_name}')
        plt.show()
    return t

def find_best_threshold(threshold, fpr, tpr):
    """
    Finds the best threshold based on maximizing the TPR while minimizing the FPR.

    Parameters:
    threshold (array-like): Threshold values.
    fpr (array-like): False Positive Rates.
    tpr (array-like): True Positive Rates.

    Returns:
    float: Best threshold value.
    """
    return threshold[np.argmax(tpr * (1 - fpr))]

def cross_validation_data(df, target_column='Tag'):
    """
    Prepares data for cross-validation.

    Parameters:
    df (DataFrame): The input DataFrame.
    target_column (str): The name of the target column.

    Returns:
    X (DataFrame), y (Series): Features and target data.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def create_classifiers(X_train, X_test, y_train, y_test, full_info=False):
    """
    Trains and evaluates classifiers.

    Parameters:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.
    y_train (Series): Training labels.
    y_test (Series): Testing labels.
    full_info (bool): If True, prints detailed information.

    Returns:
    list: Accuracy results of classifiers.
    """
    results = []
    for model_name, model in models:
        if full_info:
            print(f"\nModel: {model_name}")
            print("\nTest Train split:")
            perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train, print_info=True)
            print()
        else:
            results.append(perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train))
    return results

def perform_cross_validation(X_train, y_train, X_test, y_test, model, model_name, n_splits=10):
    """
    Performs cross-validation on the model.

    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training labels.
    X_test (DataFrame): Testing features.
    y_test (Series): Testing labels.
    model (estimator): The model to train.
    model_name (str): The name of the model.
    n_splits (int): Number of folds in KFold cross-validation.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)

    model.fit(X_train, y_train)

    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    t = create_roc(model_name, y_train, model.predict_proba(X_train)[:, 1], "cross_val", plot=False)
    y_pred_adjusted_test = (y_pred_proba_test >= t).astype(int)
    accuracy_test = accuracy_score(y_test, y_pred_adjusted_test)
    get_confusion_info(y_test, y_pred_adjusted_test)
    print(f"Accuracy: {accuracy_test:.2f}")

def perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train, print_info=False):
    """
    Trains and tests the model on the provided data.

    Parameters:
    X_test (DataFrame): Testing features.
    X_train (DataFrame): Training features.
    model (estimator): The model to train.
    model_name (str): The name of the model.
    y_test (Series): Testing labels.
    y_train (Series): Training labels.
    print_info (bool): If True, prints detailed information.

    Returns:
    float: Accuracy of the model.
    """
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    t = create_roc(model_name, y_test, y_pred_proba, "train_test", plot=print_info)
    y_pred_adjusted = (y_pred_proba >= t).astype(int)
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    if print_info:
        get_confusion_info(y_test, y_pred_adjusted)
        print(f"Accuracy: {accuracy:.2f}")
    return accuracy

def get_confusion_info(y_test, y_pred):
    """
    Prints confusion matrix information.

    Parameters:
    y_test (Series): True labels.
    y_pred (Series): Predicted labels.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")
    print(f"True Negative: {tn}")
    print(f"False Negative: {fn}")

def process_folders(base_directory):
    """
    Processes data folders and initiates model training and evaluation.

    Parameters:
    base_directory (str): Base directory containing data folders.
    """
    main_df = pd.DataFrame()
    stat_func = hp.perform_t_test

    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            main_df = dp.create_full_df(yes_folder, no_folder, folder_name, main_df)
    main_df = main_df.loc[:, ~(main_df == 0).all()]
    X_train, X_test, y_train, y_test = split_df(main_df)

    plot_features_accuracy_over_features_num(X_test, X_train, y_test, y_train, stat_func)

def plot_features_accuracy_over_features_num(X_test, X_train, y_test, y_train, statistic_func):
    """
    Plots accuracy against number of features.

    Parameters:
    X_test (DataFrame): Testing features.
    X_train (DataFrame): Training features.
    y_test (Series): Testing labels.
    y_train (Series): Training labels.
    statistic_func (function): Function to perform statistical tests.
    """
    train_df = pd.merge(y_train, X_train, left_index=True, right_index=True)
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    signi_patterns_full_data = get_signi_patterns(train_df, statistic_func)['Pattern'].unique()
    results = []
    for x in range(5, 2001, 5):
        partial_signi_patterns = signi_patterns_full_data[:x]
        if partial_signi_patterns is not None:
            significant_patterns = np.append(partial_signi_patterns, 'Tag')
            X_train_filtered = X_train.loc[:, X_train.columns.isin(significant_patterns)]
            y_train_filtered = y_train.loc[X_train_filtered.index]
            X_test_filtered = X_test.loc[:, X_test.columns.isin(significant_patterns)]
            y_test_filtered = y_test.loc[X_test_filtered.index]
        else:
            continue
        res = create_classifiers(X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered, full_info=False)
        results.append((x, res))

    for x, plot_points in results:
        for i, point in enumerate(plot_points):
            plt.plot([x + i * 5], [point], marker='o', label=f'Index {i + 1}')

    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Features')
    plt.legend()
    plt.show()

def get_signi_patterns(df, test_func, folder_name=None, prev_patterns=None):
    """
    Gets significant patterns from the DataFrame using the specified test function.

    Parameters:
    df (DataFrame): Input DataFrame.
    test_func (function): Statistical test function.
    folder_name (str): Current folder name.
    prev_patterns (DataFrame): Previous patterns DataFrame.

    Returns:
    DataFrame: Significant patterns DataFrame.
    """
    if not folder_name:
        return test_func(dp.classifiers_df_to_statistical(df))
    else:
        filtered_columns = [col for col in df.columns if col.startswith(folder_name) or col.startswith('Tag')]
        filtered_df = df[filtered_columns]

        pattern_df = filtered_df.filter(regex='_pattern$|Tag', axis=1)
        ct_cnt_df = filtered_df.filter(regex='_ct_cnt$|Tag', axis=1)
        c_loc_df = filtered_df.filter(regex='_c_loc_perc$|Tag', axis=1)

        new_pat_pattern = test_func(dp.classifiers_df_to_statistical(pattern_df), True)
        new_pat_ct_cnt = test_func(dp.classifiers_df_to_statistical(ct_cnt_df), True)
        new_pat_c_loc = test_func(dp.classifiers_df_to_statistical(c_loc_df), True)

        return pd.concat([prev_patterns, new_pat_pattern, new_pat_ct_cnt, new_pat_c_loc], ignore_index=True)

def main():
    base_directory = "/plasma_markers"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
