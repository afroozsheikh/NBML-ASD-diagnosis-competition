import argparse
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\Training Data",
        help="Path to the training data folder which contains ASD and Normal",
        required=True,
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\Primary Test Data",
        help="Path to the test data folder containing .csv files",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out",
        help="Path to the output file",
        required=True,
    )
    parser.add_argument(
        "--fc_matrix_kind",
        type=str,
        default="correlation",
        help="different kinds of functional connectivity matrices : covariance, correlation, partial correlation, tangent, precision",
        required=False,
    )
    args = parser.parse_args()
    return args


def load_data(data_dir, test_data_dir, output_dir, fc_matrix_kind):

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(
        kind=fc_matrix_kind, vectorize=False, discard_diagonal=True
    )

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(
            output_dir, "train_adjacency_" + fc_matrix_kind + ".npz"
        )
        correlation_matrices = np.load(feat_file)["a"]

        test_feat_file = os.path.join(
            output_dir, "test_adjacency_" + fc_matrix_kind + ".npz"
        )
        test_correlation_matrices = np.load(test_feat_file)["a"]

        y_target = os.path.join(output_dir, "Y_target.npz")
        y_target = np.load(y_target)["a"]

        time_series_ls = os.path.join(output_dir, "time_series.npz")
        time_series_ls = np.load(time_series_ls)["a"]

        test_time_series_ls = os.path.join(output_dir, "test_time_series.npz")
        test_time_series_ls = np.load(test_time_series_ls)["a"]

        print("Feature file found.")

    except:  # if not, extract features
        print("No feature file found. Extracting features...")
        time_series_ls = []
        test_time_series_ls = []
        correlation_matrices = []
        test_correlation_matrices = []
        y_target = []

        for (root, dirs, files) in tqdm(os.walk(data_dir, topdown=True), position=0):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)

                    ## creating y
                    if "ASD" in root:
                        y_target.append(1)
                    else:
                        y_target.append(0)

                    time_series = pd.read_csv(path)
                    time_series.drop(time_series.columns[0], axis=1, inplace=True)
                    time_series = time_series.to_numpy()
                    print(f"shape of time series : {time_series.shape}")  # (176, 110 )

                    # if fc_matrix_kind == "tangent":
                    time_series_ls.append(time_series)

                    if fc_matrix_kind != "tangent":
                        correlation_matrix = correlation_measure.fit_transform(
                            [time_series]
                        )[0]
                        correlation_matrices.append(correlation_matrix)

        if fc_matrix_kind == "tangent":
            correlation_matrices = correlation_measure.fit_transform(time_series_ls)

        for (root, dirs, files) in tqdm(
            os.walk(test_data_dir, topdown=True), position=0
        ):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)

                    time_series = pd.read_csv(path)

                    test_time_series = pd.read_csv(path)
                    test_time_series.drop(
                        test_time_series.columns[0], axis=1, inplace=True
                    )
                    test_time_series = test_time_series.to_numpy()
                    print(
                        f"shape of time series : {test_time_series.shape}"
                    )  # (176, 111)

                    # if fc_matrix_kind == "tangent":
                    test_time_series_ls.append(test_time_series)

                    if fc_matrix_kind != "tangent":
                        test_correlation_matrix = correlation_measure.fit_transform(
                            [test_time_series]
                        )[0]
                        test_correlation_matrices.append(test_correlation_matrix)

        if fc_matrix_kind == "tangent":
            test_correlation_matrices = correlation_measure.fit_transform(
                test_time_series_ls
            )

        np.savez_compressed(
            os.path.join(output_dir, "train_adjacency_" + fc_matrix_kind),
            a=correlation_matrices,
        )

        np.savez_compressed(
            os.path.join(output_dir, "test_adjacency_" + fc_matrix_kind),
            a=test_correlation_matrices,
        )

        np.savez_compressed(os.path.join(output_dir, "Y_target"), a=y_target)
        y_target = np.array(y_target)

        np.savez_compressed(
            os.path.join(output_dir, "time_series"),
            a=time_series_ls,
        )

        np.savez_compressed(
            os.path.join(output_dir, "test_time_series"),
            a=test_time_series_ls,
        )

        correlation_matrices = np.array(correlation_matrices)
        test_correlation_matrices = np.array(test_correlation_matrices)
        time_series_ls = np.array(time_series_ls)
        test_time_series_ls = np.array(test_time_series_ls)

    return (
        correlation_matrices,
        test_correlation_matrices,
        y_target,
        time_series_ls,
        test_time_series_ls,
    )


def run():
    args = parse_arguments()
    adj_mat, test_adj_mat, y_target, time_series, test_time_series = load_data(
        args.train_data_path, args.test_data_path, args.output_path, args.fc_matrix_kind
    )
    print("train adjacancy shape:", adj_mat.shape)
    print("***************************")
    print("test adjacancy shape:", test_adj_mat.shape)
    print("***************************")
    print("y target shape:", y_target.shape)
    print("***************************")
    print("time series shape:", time_series.shape)
    print("***************************")
    print("test time series shape:", test_time_series.shape)
    print("***************************")


if __name__ == "__main__":
    run()
