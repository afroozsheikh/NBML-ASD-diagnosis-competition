import argparse
import torch

from models.GATv2 import GATv2
from data.data_preparation import data_preparation
from utils.utils import count_parameters


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
        "--weights_path",
        type=str,
        help="Path to the weights file",
        required=True,
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to the folder you want to save the model results",
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
        "--heads",
        type=int,
        default=1,
        help="Number of heads used in Attention Mechanism",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="threshold use in data preprocessing section",
        required=False,
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="dropout rate used before preprocessing linear layers",
        required=False,
    )
    args = parser.parse_args()
    return args


def eval_test(model, device, dataloader):

    model.eval()
    y_pred = []

    for batch in dataloader:

        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            y_pred.append(pred)

    y_pred = torch.cat(y_pred, dim=0).float().squeeze().detach().cpu()

    return (y_pred >= 0.5).int()


def main(args):

    tag = "GATv2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    train_loader, val_loader, test_loader = data_preparation(
        adj_path=args.adj_path,
        test_adj_path=args.test_adj_path,
        y_path=args.y_path,
        time_series_path=args.time_series_path,
        test_time_series_path=args.test_time_series_path,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    model = GATv2(
        input_feat_dim=next(iter(train_loader)).x.shape[1],
        dim_shapes=[(64, 32), (32, 16), (16, 16)],
        heads=args.heads,
        num_classes=1,
        dropout_rate=args.dropout_rate,
        last_sigmoid=True,
    ).to(device)

    if args.weights_path is not None:
        model.load_weights(args.weights_path)

    count_parameters(model)

    test_y_pred = eval_test(model, device, test_loader)

    print(f"test_y_pred shape: {test_y_pred.shape}")
    print(f"test_y_pred : {test_y_pred}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
