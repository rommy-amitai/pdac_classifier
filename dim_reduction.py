import statistic_tests as hp
import data_parser as dp
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

def tsne(df, folder_name, desc):
    tags = df['Tag'].astype(int).to_numpy()
    df = df.drop(columns=['Tag'])
    mean_values = np.mean(df, axis=0)
    df_centered = df - mean_values

    if df.shape[1] > 2:
        tsne = TSNE(n_components=2,random_state=42)
    else:
        tsne = TSNE(n_components=1, random_state=42)
    df_tsne = tsne.fit_transform(df_centered)

    plt.figure(figsize=(8, 6))
    if df_tsne.shape[1] > 1:
        scatter = plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=tags, cmap='viridis', alpha=0.8)
    else:
        scatter = plt.scatter(df_tsne[:, 0], [0] * len(df_tsne), c=tags, cmap='viridis', alpha=0.8)

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
    tags = df['Tag'].astype(int).to_numpy()
    df = df.drop(columns=['Tag'])
    mean_values = np.mean(df, axis=0)
    df_centered = df - mean_values
    if df.shape[1] > 2:
        pca = PCA(n_components=2)
    else:
        pca = PCA(n_components=1)
    df_pca = pca.fit_transform(df_centered)
    plt.figure(figsize=(8, 6))

    if df_pca.shape[1] > 1:
        scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=tags, cmap='viridis', alpha=0.8)
    else:
        scatter = plt.scatter(df_pca[:, 0], [0] * len(df_pca), c=tags, cmap='viridis', alpha=0.8)

    legend_elements = [Patch(facecolor=scatter.cmap(scatter.norm(value)), edgecolor='k', label=label)
                       for value, label in zip([0, 1], ['Control', 'PDAC'])]
    plt.legend(handles=legend_elements, title='Tags', loc='lower right')

    plt.title(f'PCA Plot - {folder_name} - {desc}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.savefig(f'figs/PCA/{folder_name}_{desc}.png',
                bbox_inches='tight')  # Use bbox_inches='tight' to avoid cutting off labels
    plt.close()

def process_folders(base_directory):
    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        print(f'{folder_name}:')
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):# and folder_name=='auto-1':
            df_full = dp.complete_data_to_df(yes_folder, no_folder)
            pca(df_full, folder_name, "full data")
            tsne(df_full, folder_name, "full data")
            df_signi_pat = dp.complete_data_to_signi_pat(yes_folder, no_folder, hp.perform_ks_test)
            if df_signi_pat.shape[1] > 1:
                pca(df_signi_pat, folder_name, "signi pat data")
                tsne(df_signi_pat, folder_name, "signi pat data")

def main():
    base_directory = "/cs/usr/rommy.amitai/Desktop/project/plasma_yes_no_marker"
    process_folders(base_directory)

if __name__ == "__main__":
    main()