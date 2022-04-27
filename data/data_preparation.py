import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.centrality import (
    eigenvector,
    betweenness_centrality,
    closeness_centrality,
)
from networkx.algorithms.cluster import clustering
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\ABIDE_adjacency.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\Y_target.npz",
        help="Path to the y target",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold",
        required=False,
    )

    args = parser.parse_args()
    return args


def data_preparation(adj_path, y_path, batch_size=1, threshold=0.6):
    """Creates Data object of pytorch_geometric using graph features and edge list

    Parameters
    ----------
    adj_path : str
        path to the adjacancy matrix (.npz)

    Returns
    -------
    Data Object [torch_geometric.loader.DataLoader]
    """

    data = np.load(adj_path)
    adj_mat = data["a"]

    label = np.load(y_path)
    y_target = label["a"]
    print(y_target.shape)

    adj_mat = np.greater_equal(adj_mat, threshold).astype(int)

    data_list = []
    ## Create a graph using networkx
    for i in range(adj_mat.shape[0]):
        G = nx.from_numpy_matrix(adj_mat[i])

        ## Extract features
        features = pd.DataFrame(
            {
                "degree": dict(G.degree).values(),
                "eigen_vector_centrality": dict(
                    nx.eigenvector_centrality(G, tol=1.0e-3)
                ).values(),
                "betweenness": dict(betweenness_centrality(G)).values(),
                "closeness": dict(closeness_centrality(G)).values(),
                "clustring_coef": dict(clustering(G)).values(),
            }
        )

        X = torch.tensor(features.values)
        edge_index = torch.tensor(list(G.edges()))

        # print(y_target[i].item())
        data_list.append(Data(x=X, edge_index=edge_index.T, y=y_target[i].item()))

    ## Split Dataset
    train, val = train_test_split(
        data_list, test_size=0.2, stratify=y_target, random_state=42
    )

    train_data_loader = DataLoader(train, batch_size=batch_size)
    val_data_loader = DataLoader(val, batch_size=batch_size)

    return train_data_loader, val_data_loader


def main():
    args = parse_arguments()
    train_data_loader, val_data_loader = data_preparation(
        args.adj_path, args.y_path, args.batch_size, args.threshold
    )

    for i, data in enumerate(train_data_loader):  # every batch
        print(i, data, data.y)


if __name__ == "__main__":
    main()
