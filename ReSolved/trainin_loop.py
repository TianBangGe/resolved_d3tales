import torch
import matplotlib.pyplot as plt
import numpy as np


def train_solv(
        net,
        trainloader,
        validloader,
        device,
        initial_lr=1e-4,
        weight_decay=1e-5,
        explicit_ea=False,
        epochs=250,
):

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    criterion = torch.nn.L1Loss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        total_graphs = 0

        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            losses = []
            # We have 5 solvent columns => indices 1..5 in batch.y
            for i in range(batch.solv_dielec.shape[1]):
                out = net(
                    batch, batch.solv_dielec[:, i], batch.solv_refract[:, i])
                loss_i = criterion(out[:, 1], batch.y[:, i + 1])
                losses.append(loss_i)

            if explicit_ea == True:
                loss_ea = criterion(out[:, 0], batch.y[:, 0])
                losses.append(loss_ea*0.4)

            combined_loss = sum(losses)
            combined_loss.backward()
            optimizer.step()

            epoch_loss += combined_loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        train_avg_loss = epoch_loss / (5 * total_graphs)
        train_losses.append(train_avg_loss)

        # Validation
        net.eval()
        val_loss_sum = 0.0
        val_total_graphs = 0

        with torch.no_grad():
            for batch in validloader:
                batch = batch.to(device)
                val_losses_list = []

                for i in range(batch.solv_dielec.shape[1]):
                    out_val = net(
                        batch, batch.solv_dielec[:, i], batch.solv_refract[:, i])
                    loss_val_i = criterion(out_val[:, 1], batch.y[:, i + 1])
                    val_losses_list.append(loss_val_i)

                val_loss_sum += sum(val_losses_list).item() * batch.num_graphs
                val_total_graphs += batch.num_graphs

        val_avg_loss = val_loss_sum / (5 * val_total_graphs)
        val_losses.append(val_avg_loss)

        print(
            f"Epoch {epoch}/{epochs}: "
            f"Train MAE: {train_avg_loss:.5f} "
            f"| Val MAE : {val_avg_loss:.5f}"
        )

        # Check for best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_epoch = epoch
            torch.save(net.state_dict(), "best_model.pth")
            print(f"New best model saved, ep {best_epoch}")

    # Plot loss
    plt.plot(range(len(train_losses)), train_losses, label='Train MAE')
    plt.plot(range(len(val_losses)), val_losses, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.clf()
    return net, best_epoch
