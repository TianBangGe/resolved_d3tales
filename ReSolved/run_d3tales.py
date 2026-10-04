"""Train/Validation-only D3TaLES architecture baseline. Test stays sealed."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch_geometric
from rdkit import rdBase
from torch_geometric.loader import DataLoader

from d3tales_data import load_development_graphs
from data_utils import set_seed
from model.d3tales_model import D3TaLESModel

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data_redox"


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    error = truth - prediction
    denominator = np.square(truth - truth.mean()).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator else None,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch).flatten().cpu().tolist()
        truths = batch.y.flatten().cpu().tolist()
        for key, smiles, target, prediction in zip(batch.dataset_id, batch.smiles, truths, predictions):
            if not math.isfinite(prediction):
                raise RuntimeError(f"Non-finite validation prediction: {key}")
            rows.append(dict(dataset_id=key, canonical_smiles=smiles, reduction_potential=target,
                             prediction=prediction, residual=target - prediction))
    return metrics([r["reduction_potential"] for r in rows], [r["prediction"] for r in rows]), rows


def fit(model, trainloader, validloader, device, output, epochs, lr, weight_decay):
    output = Path(output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.L1Loss()
    best_mae, best_epoch = float("inf"), 0
    fields = ["epoch", "train_mae", "valid_mae", "valid_rmse", "valid_r2", "seconds"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            model.train()
            total_loss, count = 0.0, 0
            (output / "status.json").write_text(json.dumps(dict(
                state="training", epoch=epoch, epochs_requested=epochs, test_status="sealed",
            ), indent=2), encoding="utf-8")
            for step, batch in enumerate(trainloader, 1):
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), batch.y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite train loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                count += batch.num_graphs
                if step % 200 == 0:
                    print(f"Epoch {epoch}/{epochs} batch {step}/{len(trainloader)} "
                          f"running_train_MAE={total_loss/count:.6f}", flush=True)
            valid_metrics, _ = evaluate(model, validloader, device)
            elapsed = time.perf_counter() - started
            writer.writerow(dict(epoch=epoch, train_mae=total_loss / count,
                                 valid_mae=valid_metrics["mae"], valid_rmse=valid_metrics["rmse"],
                                 valid_r2=valid_metrics["r2"], seconds=elapsed))
            handle.flush()
            if valid_metrics["mae"] < best_mae:
                best_mae, best_epoch = valid_metrics["mae"], epoch
                torch.save(model.state_dict(), output / "best_model.pth")
            print(f"Epoch {epoch}/{epochs} train_MAE={total_loss/count:.6f} "
                  f"valid_MAE={valid_metrics['mae']:.6f} valid_RMSE={valid_metrics['rmse']:.6f} "
                  f"valid_R2={valid_metrics['r2']} seconds={elapsed:.1f}", flush=True)
            # Persist live status; only a completed run receives valid_predictions.csv.
            (output / "status.json").write_text(json.dumps(dict(
                state="training", epoch=epoch, epochs_requested=epochs,
                best_epoch=best_epoch, best_valid_mae=best_mae, test_status="sealed",
            ), indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output / "best_model.pth", map_location=device, weights_only=True))
    valid_metrics, rows = evaluate(model, validloader, device)
    with (output / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(state="complete", best_epoch=best_epoch, epochs_completed=epochs,
                   validation=valid_metrics, test_status="sealed")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    fig, ax = plt.subplots()
    ax.plot([int(r["epoch"]) for r in history], [float(r["train_mae"]) for r in history], label="Train MAE")
    ax.plot([int(r["epoch"]) for r in history], [float(r["valid_mae"]) for r in history], label="Validation MAE")
    ax.set(xlabel="Epoch", ylabel="MAE", title="D3TaLES architecture migration baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["random", "scaffold"], required=True)
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "processed/d3tales_reduction_neutral_v1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke",
                        help="default: small one-epoch smoke test; train: full development data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bond-features", choices=["legacy", "conjugation"], default="legacy",
                        # 默认不改变旧命令行为；只有显式选择才启用共轭特征。
                        help="legacy: author inputs; conjugation: replace duplicated numeric bond type with conjugation")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 1 if args.mode == "smoke" else 250
    if args.mode == "smoke" and args.epochs != 1:
        parser.error("Smoke tests run exactly one epoch; use --mode train for formal training")
    train_limit, valid_limit = (128, 64) if args.mode == "smoke" else (None, None)
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("epochs>=1, batch-size>=2, lr>0 and weight-decay>=0 required")
    split_path = DATA_ROOT / "splits" / f"{args.split}_seed42.csv"
    # --seed is the model/training seed. The split manifest always stays seed42.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; existing experiments cannot be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    torch.set_num_threads(4)
    fingerprint = {
        "data_sha256": sha256(args.data), "split_sha256": sha256(split_path),
        "graph_encoder_sha256": sha256(ROOT / "features.py"),
        "adapter_sha256": sha256(ROOT / "d3tales_data.py"),
        "graph_cache_versions": {"rdkit": rdBase.rdkitVersion,
                                 "torch": torch.__version__, "torch_geometric": torch_geometric.__version__},
        "graph_selection": {"mode": args.mode, "train_limit": train_limit, "valid_limit": valid_limit},
    }
    config = dict(
        task="D3TaLES architecture migration baseline", target="reduction_potential",
        mode=args.mode, result_scope="smoke_subset" if args.mode == "smoke" else "full_development_validation",
        split=args.split, split_file=str(split_path.resolve()), data_file=str(args.data.resolve()),
        training_seed=args.seed, split_seed=42, epochs=args.epochs, batch_size=args.batch_size,
        optimizer="AdamW", loss="MAE", lr=args.lr, weight_decay=args.weight_decay,
        num_layers=7, emb_dim=128, num_seed_points=3, heads=1,
        target_normalization="none", explicit_hydrogens=True,
        bond_features=args.bond_features,
        # 权重形状不能区分两种模式，必须保存输入语义供复现和加载时核查。
        numeric_bond_features=(["bond_type", "in_ring", "ring_size"] if args.bond_features == "legacy"
                               else ["is_conjugated", "in_ring", "ring_size"]),
        checkpoint_selection="lowest validation MAE", test_status="sealed",
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        python=platform.python_version(), torch=torch.__version__,
        torch_geometric=torch_geometric.__version__, rdkit=rdBase.rdkitVersion,
        code_sha256={name: sha256(ROOT / name) for name in [
            "model/mpnn_model.py", "model/mpnn_layer.py", "model/d3tales_model.py", "run_d3tales.py"]},
        **fingerprint,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "status.json").write_text(json.dumps(dict(state="encoding", test_status="sealed")), encoding="utf-8")
    print(f"Mode={args.mode}, split={args.split}, device={device}, bond_features={args.bond_features}, Test=SEALED", flush=True)
    if args.mode == "smoke":
        # Small local checks need no persistent graph cache.
        train, valid = load_development_graphs(args.data, split_path, train_limit, valid_limit)
    else:
        cache_dir = ROOT / "d3tales_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
        cache_path = cache_dir / f"{args.split}_{cache_key}.pt"
        if cache_path.exists():
            # Only local graph objects built by this adapter; never load external pickle caches.
            train, valid = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            train, valid = load_development_graphs(args.data, split_path)
            torch.save((train, valid), cache_path)
    config.update(train_count=len(train), valid_count=len(valid))
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Train={len(train)}, Validation={len(valid)}; no Test graphs constructed", flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = D3TaLESModel(bond_features=args.bond_features).to(device)
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        summary = fit(model, trainloader, validloader, device, args.output,
                      args.epochs, args.lr, args.weight_decay)
        summary["mode"] = args.mode
        summary["result_scope"] = config["result_scope"]
        truth = [graph.y.item() for graph in valid]
        train_targets = [graph.y.item() for graph in train]
        summary["constant_references"] = {
            "train_mean": metrics(truth, [float(np.mean(train_targets))] * len(truth)),
            "train_median": metrics(truth, [float(np.median(train_targets))] * len(truth)),
        }
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (args.output / "status.json").write_text(json.dumps(dict(
            state="failed", error=str(exc), test_status="sealed"), indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
