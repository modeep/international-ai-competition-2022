import torch
import torch.nn as nn
def real_loss(D_out, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()

    batch_size = D_out.size(0)

    labels = torch.ones(batch_size, device=device) * 0.9

    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()

    batch_size = D_out.size(0)

    labels = torch.zeros(batch_size,
                         device=device)

    loss = criterion(D_out.squeeze(), labels)
    return loss