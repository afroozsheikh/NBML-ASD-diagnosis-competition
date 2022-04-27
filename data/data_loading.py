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


def load_data(data_dir, output_dir, fc_matrix_kind):

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(
        kind=fc_matrix_kind, vectorize=False, discard_diagonal=True
    )

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(
            output_dir, "ABIDE_adjacency_" + fc_matrix_kind + ".npz"
        )
        correlation_matrices = np.load(feat_file)["a"]

        y_target = os.path.join(output_dir, "Y_target.npz")
        y_target = np.load(y_target)["a"]
        print("Feature file found.")

    except:  # if not, extract features
        print("No feature file found. Extracting features...")
        time_series_ls = []
        correlation_matrices = []
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

                    time_series = pd.read_csv(path).to_numpy()
                    print(f"shape of time series : {time_series.shape}")  # (176, 111)

                    if fc_matrix_kind == "tangent":
                        time_series_ls.append(time_series)

                    else:
                        correlation_matrix = correlation_measure.fit_transform(
                            [time_series]
                        )[0]
                        correlation_matrices.append(correlation_matrix)

        if fc_matrix_kind == "tangent":
            correlation_matrices = correlation_measure.fit_transform(time_series_ls)

        np.savez_compressed(
            os.path.join(output_dir, "ABIDE_adjacency_" + fc_matrix_kind),
            a=correlation_matrices,
        )
        correlation_matrices = np.array(correlation_matrices)

        np.savez_compressed(os.path.join(output_dir, "Y_target"), a=y_target)
        y_target = np.array(y_target)

    return correlation_matrices, y_target


def run():
    args = parse_arguments()
    adj_mat, y_target = load_data(
        args.train_data_path, args.output_path, args.fc_matrix_kind
    )
    print(adj_mat.shape)
    print("***************************")
    print(y_target.shape)
    print("***************************")


if __name__ == "__main__":
    run()
