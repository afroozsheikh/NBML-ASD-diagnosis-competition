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
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\train_adjacency_tangent.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--test_adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\test_adjacency_tangent.npz",
        help="Path to the test adjacancy matrix",
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
        "--time_series_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\time_series.npz",
        help="Path to the time series matrix",
        required=True,
    )
    parser.add_argument(
        "--test_time_series_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\test_time_series.npz",
        help="Path to the test time series matrix",
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


def data_preparation(
    adj_path,
    test_adj_path,
    y_path,
    time_series_path,
    test_time_series_path,
    batch_size=1,
    threshold=0.6,
):
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

    test_data = np.load(test_adj_path)
    test_adj_mat = test_data["a"]

    label = np.load(y_path)
    y_target = label["a"]

    time_series_data = np.load(time_series_path)
    time_series = time_series_data["a"]

    test_time_series_data = np.load(test_time_series_path)
    test_time_series = test_time_series_data["a"]

    adj_mat = np.greater_equal(adj_mat, threshold).astype(int)
    test_adj_mat = np.greater_equal(test_adj_mat, threshold).astype(int)

    data_list = []
    test_data_list = []

    ## Create a graph using networkx
    for i in range(adj_mat.shape[0]):  ## same as time_series.shape[0]
        G = nx.from_numpy_matrix(adj_mat[i])

        features = pd.DataFrame(
            {
                "degree": dict(G.degree).values(),
                "eigen_vector_centrality": dict(
                    nx.eigenvector_centrality(G, tol=1.0e-3, max_iter=10000)
                ).values(),
                "betweenness": dict(betweenness_centrality(G)).values(),
                "closeness": dict(closeness_centrality(G)).values(),
                "time_series_mean": time_series[i].mean(axis=0),
                "time_series_variance": time_series[i].var(axis=0),
                "time_series_skew": skew(time_series[i], axis=0),
                "time_series_kurtosis": kurtosis(time_series[i], axis=0),
            }
        )

        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

        X = torch.tensor(features)
        edge_index = torch.tensor(list(G.edges()))

        data_list.append(Data(x=X, edge_index=edge_index.T, y=y_target[i].item()))

    ## Split Dataset
    train, val = train_test_split(
        data_list, test_size=0.2, stratify=y_target, random_state=42
    )

    train_data_loader = DataLoader(train, batch_size=batch_size)
    val_data_loader = DataLoader(val, batch_size=batch_size)

    #### Create Test dataloader ######################################
    for i in range(test_adj_mat.shape[0]):  ## same as test_time_series.shape[0]
        G = nx.from_numpy_matrix(test_adj_mat[i])

        test_features = pd.DataFrame(
            {
                "degree": dict(G.degree).values(),
                "eigen_vector_centrality": dict(
                    nx.eigenvector_centrality(G, tol=1.0e-3)
                ).values(),
                "betweenness": dict(betweenness_centrality(G)).values(),
                "closeness": dict(closeness_centrality(G)).values(),
                "time_series_mean": test_time_series[i].mean(axis=0),
                "time_series_variance": test_time_series[i].var(axis=0),
                "time_series_skew": skew(test_time_series[i], axis=0),
                "time_series_kurtosis": kurtosis(test_time_series[i], axis=0),
            }
        )

        scaler = MinMaxScaler()
        test_features = scaler.fit_transform(test_features)

        X_test = torch.tensor(test_features)
        test_edge_index = torch.tensor(list(G.edges()))

        # print(y_target[i].item())
        test_data_list.append(Data(x=X_test, edge_index=test_edge_index.T))

    test_data_loader = DataLoader(test_data_list, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader


def main():
    args = parse_arguments()

    train_data_loader, val_data_loader, test_data_loader = data_preparation(
        args.adj_path,
        args.test_adj_path,
        args.y_path,
        args.time_series_path,
        args.test_time_series_path,
        args.batch_size,
        args.threshold,
    )

    print("************* Train Dataloader **********************")
    for i, data in enumerate(train_data_loader):  # every batch
        print(f"{i}, {data} label: {data.y}")

    print("************* Val Dataloader **********************")
    for i, data in enumerate(val_data_loader):  # every batch
        print(f"{i}, {data} label: {data.y}")

    print("************* Test Dataloader **********************")
    for i, data in enumerate(test_data_loader):  # every batch
        print(f"{i}, {data} label: {data.y}")


if __name__ == "__main__":
    main()
