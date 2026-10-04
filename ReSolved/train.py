import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from model.mpnn_model import MPNNModel
from trainin_loop import train_solv
from data_utils import SolvDataset, set_seed


def train_model(
    dataset,
    df,
    device,
    num_layers=7,
    emb_dim=128,
    magic_number=2,
    magic_number_2=3,
    magic_number_3=1,
    magic_number_4=1,
    num_seed_points=3,
    heads=1,
    dim_atoms=8,
    dim_bond=6,
    emb_dielec=2,
    emb_refract=2,
    epochs=250,
    initial_lr=1e-4,
    weight_decay=1e-5,
    seed=0,
    explicit_ea=False
):
    set_seed(seed)

    train_size = int(len(df) * 0.8)
    test_size = int((len(df) - train_size) / 2)
    val_size = len(df) - train_size - test_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_set, valid_set, test_set = random_split(
        dataset,
        [train_size, test_size, val_size],
        generator=generator
    )

    trainloader = DataLoader(train_set, batch_size=32,
                             shuffle=True, num_workers=0)
    validloader = DataLoader(valid_set, batch_size=32,
                             shuffle=False, num_workers=0)
    testloader = DataLoader(test_set, batch_size=32,
                            shuffle=False, num_workers=0)

    net = MPNNModel(
        num_layers,
        emb_dim,
        magic_number,
        magic_number_2,
        magic_number_3,
        magic_number_4,
        num_seed_points,
        heads,
        dim_atoms,
        dim_bond,
        emb_dielec,
        emb_refract,
        out_dim=1
    ).to(device)

    net, best_epoch = train_solv(
        net,
        trainloader,
        validloader,
        device,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        explicit_ea=explicit_ea,
        epochs=250
    )

    # Return split and testloader for further usage
    return net, trainloader, validloader, testloader, best_epoch
