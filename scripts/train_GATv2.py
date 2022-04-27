import argparse
from tqdm import tqdm
import torch

from models.GATv2 import GATv2
from data.data_preparation import data_preparation
from utils.utils import get_metrics, count_parameters


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\ABIDE_adjacency.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\Y_target.npz",
        help="Path to the y target",
        required=True,
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=7,
        help="Size of each node feature vector",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch",
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
        required=True,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=1,
        help="Number of heads used in Attention Mechanism",
        required=False,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning Rate used for training the model",
        required=False,
    )

    args = parser.parse_args()
    return args


def train(model, device, data_loader, optimizer, loss_fn):

    model.train()

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration", ncols=100)):

        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred.squeeze(), batch.y.float())

        loss.backward()
        optimizer.step()

    return loss.item()


def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=100)):

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return y_pred, y_true


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used Device is : {}".format(device))

    train_loader, val_loader, test_loader = data_preparation(
        args.adj_path, args.y_path, args.batch_size
    )

    model = GATv2(
        input_feat_dim=args.feat_dim,
        dim_shapes=[(128, 64), (64, 64), (64, 32)],
        heads=args.heads,
        num_layers=3,
        num_classes=1,
        dropout_p=0.005,
    ).to(device)

    # count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []

    for epoch in range(1, 1 + args.epochs):

        print("Training............")
        loss = train(model, device, train_loader, optimizer, loss_fn)

        print("Evaluating............")
        train_y_pred, train_y_true = eval(model, device, train_loader)
        train_metrics = get_metrics(train_y_pred, train_y_true, loss_fn)

        val_y_pred, val_y_true = eval(model, device, val_loader)
        val_metrics = get_metrics(val_y_pred, val_y_true, loss_fn)

        losses.append(train_metrics["loss"])

        print(
            f"Epoch: {epoch:02d} \n"
            f"Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {100 * train_metrics['acc']:.2f}%, "
            f"Val_Loss: {val_metrics['loss']:.4f}, "
            f"Val_Accuracy: {100 * val_metrics['acc']:.2f}%, "
        )
        print("===============================================================")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
