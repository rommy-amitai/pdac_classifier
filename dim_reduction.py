import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import statistic_tests as hp
import data_parser as dp

def tsne(df, folder_name, desc):
    """
    Performs t-SNE on the DataFrame and plots the result.

    Parameters:
    df (DataFrame): The input DataFrame with 'Tag' column as labels.
    folder_name (str): Name of the folder being processed.
    desc (str): Description for the plot title.

    Saves:
    A t-SNE plot as a PNG file.
    """
    tags = df['Tag'].astype(int).to_numpy()
    df = df.drop(columns=['Tag'])
    mean_values = np.mean(df, axis=0)
    df_centered = df - mean_values

    tsne = TSNE(n_components=2 if df.shape[1] > 2 else 1, random_state=42)
    df_tsne = tsne.fit_transform(df_centered)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_tsne[:, 0], df_tsne[:, 1] if df_tsne.shape[1] > 1 else [0] * len(df_tsne),
                          c=tags, cmap='viridis', alpha=0.8)

    legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(value)), edgecolor='k', label=label)
                       for value, label in zip([0, 1], ['Control', 'PDAC'])]
    plt.legend(handles=legend_elements, title='Tags', loc='lower right')

    plt.title(f't-SNE Plot - {folder_name} - {desc}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    plt.savefig(f'figs/tSNE/{folder_name}_{desc}.png', bbox_inches='tight')
    plt.close()

def pca(df, folder_name, desc):
    """
    Performs PCA on the DataFrame and plots the result.

    Parameters:
    df (DataFrame): The input DataFrame with 'Tag' column as labels.
    folder_name (str): Name of the folder being processed.
    desc (str): Description for the plot title.

    Saves:
    A PCA plot as a PNG file.
    """
    tags = df['Tag'].astype(int).to_numpy()
    df = df.drop(columns=['Tag'])
    mean_values = np.mean(df, axis=0)
    df_centered = df - mean_values

    pca = PCA(n_components=2 if df.shape[1] > 2 else 1)
    df_pca = pca.fit_transform(df_centered)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1] if df_pca.shape[1] > 1 else [0] * len(df_pca),
                          c=tags, cmap='viridis', alpha=0.8)

    legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(value)), edgecolor='k', label=label)
                       for value, label in zip([0, 1], ['Control', 'PDAC'])]
    plt.legend(handles=legend_elements, title='Tags', loc='lower right')

    plt.title(f'PCA Plot - {folder_name} - {desc}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.savefig(f'figs/PCA/{folder_name}_{desc}.png', bbox_inches='tight')
    plt.close()

def process_folders(base_directory):
    """
    Processes each folder in the base directory, performing PCA and t-SNE on the data.

    Parameters:
    base_directory (str): The base directory containing data folders.
    """
    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        print(f'{folder_name}:')
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            df_full = dp.complete_data_to_df(yes_folder, no_folder)
            pca(df_full, folder_name, "full data")
            tsne(df_full, folder_name, "full data")

            df_signi_pat = dp.complete_data_to_signi_pat(yes_folder, no_folder, hp.perform_ks_test)
            if df_signi_pat.shape[1] > 1:
                pca(df_signi_pat, folder_name, "signi pat data")
                tsne(df_signi_pat, folder_name, "signi pat data")

def main():
    """
    Main function to process folders in the specified base directory.
    """
    base_directory = "/plasma_markers"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
