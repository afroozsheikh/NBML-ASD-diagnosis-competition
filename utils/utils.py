import torch
from torchmetrics import Accuracy
from prettytable import PrettyTable


def get_metrics(y_pred, y_true, loss_fn):

    y_pred_int = (y_pred >= 0.5).type(torch.int32)

    metrics = {}
    acc = Accuracy()

    metrics["acc"] = acc(y_pred_int, y_true)
    metrics["loss"] = loss_fn(y_pred, y_true.float())

    return metrics


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters shape", "Total Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_shape = parameter.shape
            params = parameter.numel()
            table.add_row([name, param_shape, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
