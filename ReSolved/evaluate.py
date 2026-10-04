import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from model.mpnn_model import MPNNModel
from data_utils import set_seed


def eval_model(
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
    seed=0,
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

    # Return split and testloader for further usage
    return net, trainloader, validloader, testloader


def process_dataset(net, loader, dataset_name, device, solvent_count=5):
    """
    Evaluate across each of the 5 solvents, generate scatter plots and CSV
    """
    net.eval()
    all_data = []

    for solv_i in range(solvent_count):
        predictions = []
        real = []
        smiles_list = []

        for batch in loader:
            batch = batch.to(device)
            out = net(
                batch, batch.solv_dielec[:, solv_i], batch.solv_refract[:, solv_i])
            pred = out[:, 1].detach().cpu().numpy()
            gt = batch.y[:, solv_i + 1].detach().cpu().numpy()

            predictions.extend(pred)
            real.extend(gt)
            smiles_list.extend(batch.smiles)

        # Plot
        r2_val = r2_score(real, predictions)
        mae_val = mean_absolute_error(real, predictions)

        plt.scatter(real, predictions, marker='o',
                    c='b', edgecolors='k', alpha=0.2)
        plt.plot(
            np.linspace(min(real), max(real)),
            np.linspace(min(real), max(real)),
            color='r'
        )
        plt.xlabel('DFT')
        plt.ylabel('MPNN')
        plt.title(f"{dataset_name}: Solvent {solv_i+1}")
        plt.text(
            0.05, 0.95,
            f'R² = {r2_val:.3f}\nMAE = {mae_val:.3f}',
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top'
        )
        plt.savefig(f"{dataset_name}_solvent_{solv_i+1}.png")
        plt.clf()

        # Save for CSV
        for idx, smile in enumerate(smiles_list):
            if idx >= len(all_data):
                all_data.append({'SMILES': smile})
            all_data[idx][f'SolventTruth{solv_i+1}'] = real[idx]
            all_data[idx][f'Predicted{solv_i+1}'] = predictions[idx]

    # Per-molecule MAE across all solvents
    for row in all_data:
        gt_vals = [row[f'SolventTruth{i+1}'] for i in range(solvent_count)]
        pred_vals = [row[f'Predicted{i+1}'] for i in range(solvent_count)]
        row['MAE'] = mean_absolute_error(gt_vals, pred_vals)

    df_out = pd.DataFrame(all_data)
    df_out.to_csv(f"{dataset_name}_prediction.csv", index=False)


def process_dataset_ea(net, loader, dataset_name, device):
    """
    Evaluate only the EA channel (the first output)
    """
    net.eval()
    all_data = []
    real_vals = []
    pred_vals = []

    for batch in loader:
        batch = batch.to(device)
        out = net(batch, batch.solv_dielec[:, 0], batch.solv_refract[:, 0])
        pred = out[:, 0].detach().cpu().numpy()
        gt = batch.y[:, 0].detach().cpu().numpy()

        real_vals.extend(gt)
        pred_vals.extend(pred)

        for s, g, p in zip(batch.smiles, gt, pred):
            all_data.append({
                'SMILES': s,
                'EA_True': g,
                'EA_Pred': p
            })

    r2_val = r2_score(real_vals, pred_vals)
    mae_val = mean_absolute_error(real_vals, pred_vals)

    # Plot
    plt.scatter(real_vals, pred_vals, marker='o',
                c='b', edgecolors='k', alpha=0.2)
    plt.plot(
        np.linspace(min(real_vals), max(real_vals)),
        np.linspace(min(real_vals), max(real_vals)),
        color='r'
    )
    plt.xlabel("DFT (EA)")
    plt.ylabel("MPNN (EA)")
    plt.title(f"{dataset_name}: EA Predictions")
    plt.text(
        0.05, 0.95,
        f'R2 = {r2_val:.3f}\nMAE = {mae_val:.3f}',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top'
    )
    plt.savefig(f"{dataset_name}_ea.png")
    plt.clf()

    # Save CSV
    df_ea = pd.DataFrame(all_data)
    df_ea.to_csv(f"{dataset_name}_ea_prediction.csv", index=False)
