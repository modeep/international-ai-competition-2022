import torch
import torch.nn as nn
def real_loss(D_out, device='cpu'):
    # initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()

    # batch size
    batch_size = D_out.size(0)

    # labels will be used when calculating the losses
    # real labels = 1 and lable smoothing => 0.9
    labels = torch.ones(batch_size, device=device) * 0.9

    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out, device='cpu'):
    # initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()

    # batch size
    batch_size = D_out.size(0)

    # labels will be used when calculating the losses
    # fake labels = 0
    labels = torch.zeros(batch_size,
                         device=device)

    loss = criterion(D_out.squeeze(), labels)
    return loss