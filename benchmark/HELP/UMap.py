import pandas as pd
import numpy as np
import umap
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EmbeddingsModel import EmbeddingsModel
import torch
import argparse

def get_embeddings_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def visualize_umap(vis_nn):
    dataset = 'HealthApp'
    folder_name = 'wordcount'

    input_dir = f'../../full_dataset/{dataset}/{dataset}_full.log_structured.csv'
    df = pd.read_csv(input_dir)
    labels = df['EventId'].values[:1000]

    # load data
    logVectors = get_embeddings_from_file(f'openai_embeddings/{dataset}_{folder_name}_text-embedding-3-small/{0}.pkl')[:1000]
    print("got vectors ", len(logVectors))

    if vis_nn:
        nn = EmbeddingsModel(1536, 750, 0)
        nn.load_state_dict(torch.load(f'nn_models_hyp_scan_wc/{dataset}/{dataset}_out-750_bs_2048_max_epochs_50_dropout-0.2_lr-0.0005.pth'))
        nn.eval()
        with torch.inference_mode():
            logVectors = np.array(logVectors, dtype=np.float32)
            logVectors = torch.tensor(logVectors)
            output = nn(logVectors)
            logVectors = output.detach()
            logVectors = logVectors / torch.linalg.vector_norm(logVectors, dim=1, keepdim=True)
            logVectors = logVectors.cpu().numpy()

    # Perform UMAP dimensionality reduction to 3D
    umap_model = umap.UMAP(n_neighbors=20, min_dist=0.08, n_components=3, random_state=42)
    umap_embeddings = umap_model.fit_transform(logVectors)

    # Create a DataFrame for the embeddings and labels
    embeddings_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    embeddings_df['label'] = labels

    # Plot the UMAP embeddings with cluster labels in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings_df['UMAP1'], embeddings_df['UMAP2'], embeddings_df['UMAP3'], c=embeddings_df['label'].astype('category').cat.codes, cmap='Spectral', s=20, alpha=0.4) # change opacity of dots to 0.8
    if vis_nn:
        plt.title(f'UMAP 3D projection of the {folder_name} OpenAI embeddings with NN')
    else:
        plt.title(f'UMAP 3D projection of the {folder_name} OpenAI embeddings')

    if vis_nn:
        plt.savefig('umap_3d_projection_nn.png', dpi=300)
    else:
        plt.savefig('umap_3d_projection.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    # parse for vis_nn
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', action='store_true', default=False, help='Enable or disable nn flag')
    args = parser.parse_args()
    vis_nn = args.nn
    print(vis_nn)
    visualize_umap(vis_nn)
