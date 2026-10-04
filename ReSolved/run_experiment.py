import torch
import pandas as pd

from data_utils import SolvDataset, set_seed
from train import train_model
from evaluate import (
    process_dataset,
    process_dataset_ea
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    # Load data from CSV
    df = pd.read_csv("resolved.csv")

    # Build dataset
    dataset = SolvDataset(
        df["smiles"],
        df["EA"],
        df["RP_ACN"],
        df["RP_H2O"],
        df["RP_THF"],
        df["RP_DMSO"],
        df["RP_DMF"],
        device
    )

    # Train the model
    model, trainloader, validloader, testloader, best_epoch = train_model(
        dataset=dataset,
        df=df,
        device=device,
        explicit_ea=False
    )
    print(f"Best model at epoch {best_epoch}")

    # Loading the best model for inference
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Evaluate train, test, validation for solvent properties
    with torch.no_grad():
        process_dataset(model, trainloader, "train", device=device)
        process_dataset(model, validloader, "validation", device=device)
        process_dataset(model, testloader, "test", device=device)

        # Evaluate EA channel
        process_dataset_ea(model, trainloader, "train_ea", device=device)
        process_dataset_ea(model, validloader, "validation_ea", device=device)
        process_dataset_ea(model, testloader, "test_ea", device=device)


if __name__ == "__main__":
    main()
