"""Train/Validation diagnostics; no Test evaluation or baseline modification."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from d3tales_data import molecule_graph
from data_utils import set_seed
from model.d3tales_model import D3TaLESModel

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data_redox"
OUTPUT = ROOT / "results_d3tales" / "diagnostics"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def error_summary(rows):
    """按原始目标尺度计算误差；bias 为真值减预测，正值表示低估。"""
    if not rows:
        return {"n": 0}
    target = np.array([float(r["reduction_potential"]) for r in rows])
    prediction = np.array([float(r["prediction"]) for r in rows])
    error = target - prediction
    return dict(n=len(rows), mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())),
                bias=float(error.mean()), target_mean=float(target.mean()),
                prediction_mean=float(prediction.mean()), target_std=float(target.std()),
                prediction_std=float(prediction.std()), sse=float((error**2).sum()))


def residual_audit(metadata, manifests):
    """只分析既有 Validation 预测，结构组允许重叠，不作因果归因。"""
    result = {}
    patterns = {
        "nitrile": Chem.MolFromSmarts("[C]#[N]"),
        "carbonyl": Chem.MolFromSmarts("[CX3]=[OX1]"),
        "imide": Chem.MolFromSmarts("[NX3]([CX3](=[OX1]))[CX3](=[OX1])"),
        "diaryl_amine": Chem.MolFromSmarts("[NX3]([c])[c]"),
    }
    for split in ("random", "scaffold"):
        folder = ROOT / "results_d3tales/local_full_baseline" / f"{split}_seed42_250epochs"
        rows = csv_rows(folder / "valid_predictions.csv")
        expected = {key for key, row in manifests[split].items() if row["split"] == "valid"}
        # 同时核查集合与行数，防止混入 Test、缺失或重复的预测记录。
        assert {r["dataset_id"] for r in rows} == expected and len(rows) == len(expected)
        source_groups, structure_groups = defaultdict(list), defaultdict(list)
        for row in rows:
            source_groups[metadata[row["dataset_id"]]["source_groups"]].append(row)
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            for name, pattern in patterns.items():
                if mol.HasSubstructMatch(pattern):
                    structure_groups[name].append(row)
            if any(a.GetFormalCharge() for a in mol.GetAtoms()):
                structure_groups["internal_formal_charges"].append(row)
            if len(Chem.GetMolFrags(mol)) > 1:
                structure_groups["multiple_fragments"].append(row)
        bins = {}
        for lo, hi in ((3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)):
            bins[f"[{lo},{hi})"] = error_summary([r for r in rows if lo <= float(r["reduction_potential"]) < hi])
        training_counts = defaultdict(int)
        # Parse development labels only. Never inspect that split's Test target.
        for key, assignment in manifests[split].items():
            if assignment["split"] != "train":
                continue
            y = float(metadata[key]["reduction_potential"])
            for lo, hi in ((3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12)):
                if lo <= y < hi:
                    training_counts[f"[{lo},{hi})"] += 1
        result[split] = dict(overall=error_summary(rows), target_bins=bins,
                             train_target_bin_counts=dict(training_counts),
                             sources={name: error_summary(rr) for name, rr in source_groups.items()},
                             structure_groups={name: error_summary(rr) for name, rr in structure_groups.items()})
    return result


def raw_label_audit(metadata, manifests):
    """先排除任一划分中的 Test 分子，再转换开发记录的原始数值标签。"""
    # Restrict raw-label inspection to molecules outside BOTH Test manifests.
    eligible = {key for key in metadata if all(m[key]["split"] != "test" for m in manifests.values())}
    original_ids = {raw_id for key in eligible for raw_id in metadata[key]["original_ids"].split("|")}
    groups = defaultdict(list)
    compared, same_solv = 0, 0
    with (DATA / "raw/d3tales_public.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["_id"] not in original_ids:
                continue
            if row["reduction_potential"]:
                compared += 1
                same_solv += row["reduction_potential"] == row["solv_reduction_potential"]
            if row["reduction_potential"] and row["adiabatic_electron_affinity"]:
                groups[row["source_group"]].append(float(row["reduction_potential"]) + float(row["adiabatic_electron_affinity"]))
    return dict(scope="outside both Test manifests", n_raw_with_reduction=compared,
                reduction_and_solv_fields_equal=same_solv,
                reduction_plus_adiabatic_ea={k: dict(n=len(v), median=statistics.median(v),
                                                    min=min(v), max=max(v)) for k, v in groups.items()})


def overfit_probe(metadata, manifest):
    """固定抽取 32 个 Train 分子拟合 300 次，不用 Validation 选模型。"""
    set_seed(42)
    torch.set_num_threads(4)
    ids = sorted(key for key, row in manifest.items() if row["split"] == "train")
    chosen = sorted(np.random.default_rng(42).choice(ids, size=32, replace=False).tolist())
    graphs = [molecule_graph(key, metadata[key]["canonical_smiles"], float(metadata[key]["reduction_potential"])) for key in chosen]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = Batch.from_data_list(graphs).to(device)
    model = D3TaLESModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.L1Loss()
    records = []
    # 32 Train molecules, 300 gradient steps, no Validation/model selection/Test.
    for step in range(301):
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                evaluation_mae = loss_fn(model(batch), batch.y).item()
            model.train()
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            with torch.no_grad():
                training_mode_mae = loss_fn(model(batch), batch.y).item()
                # 训练模式前向会更新 BatchNorm 统计量；测量结束后恢复缓冲区，
                # 避免诊断自身改变训练过程。参数与梯度在这里不作更新。
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
            records.append(dict(step=step, train_mode_mae=training_mode_mae, eval_mode_train_mae=evaluation_mae))
            print(f"Train-only probe step={step}/300 train_mode_MAE={training_mode_mae:.6f} eval_mode_MAE={evaluation_mae:.6f}", flush=True)
        if step == 300:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), batch.y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
        changed = batch.clone()
        # 仅翻转图副本的共轭列，检查旧模型是否实际使用该输入。
        changed.edge_attr[:, 1] = 1 - changed.edge_attr[:, 1]
        conjugation_effect = (model(changed) - prediction).abs().max().item()
    with (OUTPUT / "train_only_fit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return dict(n_train=32, steps=300, device=str(device), dataset_ids=chosen,
                lr=1e-4, weight_decay=1e-5, final=records[-1],
                best_observed_eval_mode_train_mae=min(r["eval_mode_train_mae"] for r in records),
                conjugation_column_flip_max_prediction_difference=conjugation_effect,
                checkpoint_saved=False, test_status="sealed")


def main():
    # 本目录仅保存可重算的诊断结果，不覆盖正式训练输出或保存新权重。
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = {split: {r["dataset_id"]: r for r in csv_rows(DATA / f"splits/{split}_seed42.csv")}
                 for split in ("random", "scaffold")}
    # Metadata contains raw strings; only selected development rows are converted.
    metadata = {r["dataset_id"]: r for r in csv_rows(DATA / "processed/d3tales_reduction_neutral_v1.csv")}
    result = dict(test_status="sealed", residuals=residual_audit(metadata, manifests),
                  raw_label_checks=raw_label_audit(metadata, manifests))
    (OUTPUT / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["train_only_fit"] = overfit_probe(metadata, manifests["random"])
    (OUTPUT / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Diagnostics saved:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
