import sklearn
import statistic_tests as hp
import data_parser as dp
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb

RANDOM_STATE= 42

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Support Vector Machine _ poly', SVC(kernel='poly', probability=True)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('XGBoost', xgb.XGBClassifier())  # Adding XGBoost
]

def split_df(df, test_size=0.2, random_state=RANDOM_STATE):
    y = df.iloc[:, 0]  # Assuming the first column is the target
    X = df.iloc[:, 1:]  # Assuming the remaining columns are features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_roc(model_name, y, y_pred_proba, desc, plot=None):
    auc = roc_auc_score(y, y_pred_proba)
    fpr, tpr, threshold = roc_curve(y, y_pred_proba)
    t = find_best_threshold(threshold, fpr, tpr)
    if plot!= False:
        plt.plot(fpr, tpr, label="AUC=" + str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.title(f'{desc.upper()} | ROC Curve for {model_name}')
        plt.show()
    return t


def find_best_threshold(threshould, fpr, tpr):
   t = threshould[np.argmax(tpr*(1-fpr))]
   return t

def cross_validation_data(df, target_column='Tag'):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def create_classifiers(X_train, X_test, y_train, y_test, full_info=None):
    res_over_all_methods=[]
    for model_name, model in models:
        # print("\nCross Validation:")
        # #perform_cross_validation(X,y, model, model_name)
        # perform_cross_validation(X_train, y_train,X_test, y_test, model, model_name)
        if full_info==True:
            print(f"\nModel: {model_name}")
            print("\nTest Train split:")
            perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train)
            print()
        else:
            res_over_all_methods.append(perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train, False))
    return res_over_all_methods


def perform_cross_validation(X_train, y_train, X_test, y_test, model, model_name, n_splits=10):
    # Perform cross-validation on the training set
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fit model on train fold
        model.fit(X_train_fold, y_train_fold)

    # Fit model on the entire training set
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate threshold and adjust predictions based on the training set
    t = create_roc(model_name, y_train, model.predict_proba(X_train)[:, 1], "cross_val")
    print('after thresh adjust')
    y_pred_adjusted_test = (y_pred_proba_test >= t).astype(int)
    # Calculate accuracy and AUC on test set
    accuracy_test = accuracy_score(y_test, y_pred_adjusted_test)
    get_confusion_info(y_test, y_pred_adjusted_test)
    print(f"Accuracy: {accuracy_test:.2f}")
    #auc_test = roc_auc_score(y_test, y_pred_proba_test)

    # Print confusion matrix and accuracy for test set
    # print("Test Set Metrics:")
    # print(f"Test set accuracy: {accuracy_test:.2f}")
    # print(f"Test set AUC: {auc_test:.2f}")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred_adjusted_test))

def perform_cross_validation_old(X, y, model, model_name, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    # scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    # mean_accuracy = scores.mean()
    # print('without thresh adjust')
    # print(f"Accuracy: {mean_accuracy:.2f}")


    # SEND ONLY TEST SET
    # feature selection on train
    # stratified
    # feature analysis
    # choose one classifier that works best
    # check what happens to missing samples (one marker)
    # show which markers are stronger than others
    # synthetic combinations, try with current models

    y_pred_proba = cross_val_predict(model, X, y, cv=kfold, method='predict_proba')[:, 1]
    t = create_roc(model_name, y, y_pred_proba, "cross_val")

    print('after thresh adjust')
    y_pred_adjusted = (y_pred_proba >= t).astype(int)
    accuracy = accuracy_score(y, y_pred_adjusted)
    get_confusion_info(y, y_pred_adjusted)
    print(f"Accuracy: {accuracy:.2f}")

def perform_train_test_classification(X_test, X_train, model, model_name, y_test, y_train, print=None):
    model.fit(X_train, y_train)
    # print('without thresh adjust')
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    t = create_roc(model_name, y_test, y_pred_proba, "train_test", print)
    #print('after thresh adjust')
    y_pred_adjusted = (y_pred_proba >= t).astype(int)
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    if print==True:
        get_confusion_info(y_test, y_pred_adjusted)
        print(f"Accuracy: {accuracy:.2f}")
    return accuracy
    # get_confusion_info(y_test, y_pred_adjusted)


def get_confusion_info(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"True Positive: {tp:.2f}")
    print(f"False Positive: {fp:.2f}")
    print(f"True Negative: {tn:.2f}")
    print(f"False Negative: {fn:.2f}")


def process_folders(base_directory):
    # Iterate through folders in the base directory
    main_df = pd.DataFrame()
    #main_df_ = pd.DataFrame()
    #signi_patterns= pd.DataFrame()
    stat_func= hp.perform_t_test


    for folder_name in os.listdir(base_directory):
        # Construct paths for "yes" and "no" folders within each folder
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        # Check if both "yes" and "no" folders exist
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            # Process the data in the current pair of "yes" and "no" folders
            #main_df_, signi_patterns= create_full_df_sep_markers(yes_folder, no_folder, folder_name, main_df_, hp.perform_t_test, signi_patterns)
            main_df = create_full_df(yes_folder, no_folder, folder_name, main_df)
    main_df = main_df.loc[:, ~(main_df == 0).all()]
    #print(signi_patterns['Pattern'].unique())
    #return
    X_train, X_test, y_train, y_test = split_df(main_df)

    #create_classification_over_signi_patterns(X_test, X_train, y_test, y_train, stat_func, base_directory)
    plot_features_accuracy_over_features_num(X_test, X_train, y_test, y_train, stat_func)

    # print()
    # print("all data")
    # create_classifiers(X_train, X_test, y_train, y_test)

def plot_features_accuracy_over_features_num(X_test, X_train,
                                             y_test, y_train,
                                             statistic_func):
    train_df = pd.merge(y_train, X_train, left_index=True, right_index=True, suffixes=('', '_drop'))
    train_df = train_df.loc[:, ~train_df.columns.str.endswith('_drop')]
    signi_patterns_full_data = get_signi_patterns(train_df, statistic_func)['Pattern'].unique()
    results = []
    for x in range(5, 2001, 5):
        partial_signi_patterns= signi_patterns_full_data[:x]
        if partial_signi_patterns is not None:
            significant_patterns = np.append(partial_signi_patterns, 'Tag')
            X_train_filtered = X_train.loc[:, X_train.columns.isin(significant_patterns)]
            y_train_filtered = y_train.loc[X_train_filtered.index]
            X_test_filtered = X_test.loc[:, X_test.columns.isin(significant_patterns)]
            y_test_filtered = y_test.loc[X_test_filtered.index]
        else:
            continue
        res=create_classifiers(X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered, False)
        results.append((x, res))

    for x, plot_points in results:
        for i, point in enumerate(plot_points):
            plt.plot([x + i * 5], [point], marker='o', label=f'Index {i + 1}')

    # Adding labels and title
    plt.xlabel('Num of features')
    plt.ylabel('Accuracy')
    plt.title('Individual Plots from Results')
    plt.legend()

    # Display the plots
    plt.show()


def create_classification_over_signi_patterns(X_test, X_train,
                                              y_test, y_train,
                                              base_directory,
                                              statistic_func):
    train_df = pd.merge(y_train, X_train, left_index=True, right_index=True, suffixes=('', '_drop'))
    train_df = train_df.loc[:, ~train_df.columns.str.endswith('_drop')]
    signi_patterns_per_marker = pd.DataFrame()
    for folder_name in os.listdir(base_directory):
        signi_patterns_per_marker = get_signi_patterns(train_df,
                                                       statistic_func,
                                                       folder_name,
                                                       signi_patterns_per_marker)
    signi_patterns_per_marker = signi_patterns_per_marker['Pattern'].unique()
    print("only significant patterns")
    print(signi_patterns_per_marker)
    if signi_patterns_per_marker is not None:
        significant_patterns = np.append(signi_patterns_per_marker, 'Tag')
        X_train_filtered = X_train.loc[:, X_train.columns.isin(significant_patterns)]
        y_train_filtered = y_train.loc[X_train_filtered.index]
        X_test_filtered = X_test.loc[:, X_test.columns.isin(significant_patterns)]
        y_test_filtered = y_test.loc[X_test_filtered.index]
    else:
        return
    create_classifiers(X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered, True)


def get_signi_patterns(df, test_func, folder_name=None, prev_patterns=None):
    if not folder_name:
        return test_func(dp.classifiers_df_to_statistical(df))
    else:
        filtered_columns = [col for col in df.columns if col.startswith(folder_name) or col.startswith('Tag')]
        filtered_df = df[filtered_columns]
        #new_patterns= test_func(dp.classifiers_df_to_statistical(filtered_df))

        pattern_df = filtered_df.filter(regex='_pattern$|Tag', axis=1)
        ct_cnt_df = filtered_df.filter(regex='_ct_cnt$|Tag', axis=1)
        c_loc_df = filtered_df.filter(regex='_c_loc_perc$|Tag', axis=1)

        new_pat_pattern= test_func(dp.classifiers_df_to_statistical(pattern_df), True)
        new_pat_ct_cnt = test_func(dp.classifiers_df_to_statistical(ct_cnt_df), True)
        new_pat_c_loc = test_func(dp.classifiers_df_to_statistical(c_loc_df), True)

        return pd.concat([prev_patterns, new_pat_pattern, new_pat_ct_cnt, new_pat_c_loc], ignore_index=True)


def create_full_df_sep_markers(yes_path, no_path, folder_name, main_df, test_func, signi_patterns):
    pattern_df =dp.create_df_for_classification(yes_path, no_path)
    ct_cnt_df = dp.single_pat_to_ct_cnt(pattern_df)
    c_loc_df= dp.single_pat_to_c_loc_perc(pattern_df)

    pattern_df = dp.scale_df(pattern_df)
    ct_cnt_df = dp.scale_df(ct_cnt_df)
    c_loc_df = dp.scale_df(c_loc_df)

    cnt_df_sel = ct_cnt_df.iloc[:, 1:]
    c_loc_df_sel = c_loc_df.iloc[:, 1:]
    local_df = pd.concat([pattern_df,
                         cnt_df_sel,
                         c_loc_df_sel], axis=1)

    signi_single_patterns_df = test_func(dp.classifiers_df_to_statistical(pattern_df))
    signi_ct_cnt_patterns_df = test_func(dp.classifiers_df_to_statistical(ct_cnt_df))
    signi_c_loc_patterns_df = test_func(dp.classifiers_df_to_statistical(c_loc_df))
    signi_patterns_df = pd.concat([signi_single_patterns_df,
                                   signi_ct_cnt_patterns_df,
                                   signi_c_loc_patterns_df], ignore_index=True)

    signi_patterns_df['Pattern'] = f'{folder_name}_' + signi_patterns_df['Pattern'].astype(str)
    new_column_names_local = [(f'{folder_name}_' + col) if col != 'Tag' else col for col in local_df.columns]
    local_df.columns = new_column_names_local

    if not signi_patterns.empty:
        signi_patterns_df = pd.concat([signi_patterns_df, signi_patterns], ignore_index=True)
    if not main_df.empty:
        local_df= pd.merge(local_df, main_df, left_index=True, right_index=True, suffixes=('', '_drop'))
        local_df = local_df.loc[:, ~local_df.columns.str.endswith('_drop')]

    return local_df, signi_patterns_df

def create_full_df(yes_path, no_path, folder_name, main_df):
    pattern_df =dp.create_df_for_classification(yes_path, no_path)
    ct_cnt_df = dp.single_pat_to_ct_cnt(pattern_df)
    c_loc_df= dp.single_pat_to_c_loc_perc(pattern_df)

    pattern_df.columns = [f'{col}_pattern' if 'Tag' not in col else col for col in pattern_df.columns]
    ct_cnt_df.columns = [f'{col}_ct_cnt' if 'Tag' not in col else col for col in ct_cnt_df.columns]
    c_loc_df.columns = [f'{col}_c_loc_perc' if 'Tag' not in col else col for col in c_loc_df.columns]

    pattern_df = dp.scale_df(pattern_df)
    ct_cnt_df = dp.scale_df(ct_cnt_df)
    c_loc_df = dp.scale_df(c_loc_df)

    cnt_df_sel = ct_cnt_df.iloc[:, 1:]
    c_loc_df_sel = c_loc_df.iloc[:, 1:]
    local_df = pd.concat([pattern_df,
                         cnt_df_sel,
                         c_loc_df_sel], axis=1)

    new_column_names_local = [(f'{folder_name}_' + col) if col != 'Tag' else col for col in local_df.columns]
    local_df.columns = new_column_names_local

    if not main_df.empty:
        local_df= pd.merge(local_df, main_df, left_index=True, right_index=True, suffixes=('', '_drop'))
        local_df = local_df.loc[:, ~local_df.columns.str.endswith('_drop')]

    return local_df


def main():
    base_directory = "/cs/usr/rommy.amitai/Desktop/project/plasma_yes_no_marker"
    process_folders(base_directory)

if __name__ == "__main__":
    main()