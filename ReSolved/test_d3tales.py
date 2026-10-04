"""Behavioral checks for the single-target migration and sealed-test protocol."""
import csv
import importlib.util
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
from torch_geometric.data import Batch


@contextmanager
def fixture_directory():
    # Python's mkdtemp(mode=0o700) is incompatible with this Windows sandbox.
    root = Path(__file__).parent / ("test_fixture_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        for file in root.iterdir():
            file.unlink()
        root.rmdir()


class D3TaLESTests(unittest.TestCase):
    def require_module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f"Missing D3TaLES adapter: {name}")

    def test_graph_encoding_matches_author_and_preserves_target_sign(self):
        self.require_module("d3tales_data")
        from d3tales_data import molecule_graph
        from data_utils import smi_to_pyg
        graph = molecule_graph("example", "CC=O", 6.25)
        original = smi_to_pyg("CC=O", 0, 0, 0, 0, 0, 0, torch.device("cpu"))
        for field in ("x", "edge_index", "edge_attr"):
            self.assertTrue(torch.equal(graph[field], original[field]), field)
        self.assertEqual(graph.num_nodes, 7)
        self.assertEqual(graph.y.tolist(), [[6.25]])
        self.assertEqual(graph.dataset_id, "example")

    def test_single_output_is_batch_invariant_and_backpropagates(self):
        self.require_module("d3tales_data")
        self.require_module("model.d3tales_model")
        from d3tales_data import molecule_graph
        from model.d3tales_model import D3TaLESModel
        torch.manual_seed(42)
        graphs = [molecule_graph("a", "CC=O", 6.25), molecule_graph("b", "c1ccccc1", 5.5)]
        model = D3TaLESModel(num_layers=2, emb_dim=16)
        batch = Batch.from_data_list(graphs)
        model.eval()
        predictions = model(batch)
        self.assertEqual(tuple(predictions.shape), (2, 1))
        self.assertTrue(torch.isfinite(predictions).all())
        single = model(Batch.from_data_list(graphs[:1]))
        torch.testing.assert_close(single[0], predictions[0], atol=1e-5, rtol=1e-5)
        model.train()
        torch.nn.functional.l1_loss(model(batch), batch.y).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def write_fixture(self, root):
        data = root / "data.csv"
        split = root / "split.csv"
        with data.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset_id", "canonical_smiles", "reduction_potential", "groundState_charge"])
            writer.writerows([["a", "CC=O", 6.25, 0], ["b", "CCO", 5.5, 0],
                              ["sealed", "NOT_A_SMILES", "SEALED_NOT_A_NUMBER", 0]])
        with split.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset_id", "canonical_smiles", "split"])
            writer.writerows([["b", "CCO", "valid"], ["sealed", "NOT_A_SMILES", "test"], ["a", "CC=O", "train"]])
        return data, split

    def test_id_alignment_and_test_rows_are_never_parsed_or_featurized(self):
        self.require_module("d3tales_data")
        from d3tales_data import load_development_graphs
        with fixture_directory() as directory:
            data, split = self.write_fixture(Path(directory))
            train, valid = load_development_graphs(data, split)
            self.assertEqual([g.dataset_id for g in train], ["a"])
            self.assertEqual([g.dataset_id for g in valid], ["b"])
            self.assertEqual(train[0].y.item(), 6.25)
            self.assertEqual(valid[0].y.item(), 5.5)

    def test_duplicate_split_ids_are_rejected(self):
        self.require_module("d3tales_data")
        from d3tales_data import load_development_graphs
        with fixture_directory() as directory:
            data, split = self.write_fixture(Path(directory))
            with split.open("a", encoding="utf-8") as handle:
                handle.write("a,CC=O,valid\n")
            with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                load_development_graphs(data, split)

    def test_smoke_limits_skip_unused_development_graphs_before_parsing(self):
        self.require_module("d3tales_data")
        from d3tales_data import load_development_graphs
        with fixture_directory() as directory:
            data, split = self.write_fixture(Path(directory))
            with data.open("a", encoding="utf-8") as handle:
                handle.write("unused,BAD_UNUSED_SMILES,NOT_A_NUMBER,0\n")
            with split.open("a", encoding="utf-8") as handle:
                handle.write("unused,BAD_UNUSED_SMILES,train\n")
            train, valid = load_development_graphs(data, split, train_limit=1, valid_limit=1)
            self.assertEqual([g.dataset_id for g in train], ["a"])
            self.assertEqual([g.dataset_id for g in valid], ["b"])

    def test_invalid_development_smiles_fail_without_silently_dropping_ids(self):
        self.require_module("d3tales_data")
        from d3tales_data import molecule_graph
        with self.assertRaisesRegex(ValueError, "broken"):
            molecule_graph("broken", "NOT_A_SMILES", 6.0)

    def test_training_writes_only_validation_predictions_and_best_checkpoint(self):
        self.require_module("run_d3tales")
        from run_d3tales import fit
        from d3tales_data import molecule_graph
        from model.d3tales_model import D3TaLESModel
        from torch_geometric.loader import DataLoader
        with fixture_directory() as root:
            graphs = [molecule_graph("a", "CC=O", 6.25), molecule_graph("b", "CCO", 5.5)]
            loader = DataLoader(graphs, batch_size=2, shuffle=False)
            summary = fit(D3TaLESModel(num_layers=1, emb_dim=16), loader, loader,
                          torch.device("cpu"), root, epochs=2, lr=1e-4, weight_decay=1e-5)
            self.assertEqual(summary["test_status"], "sealed")
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertTrue((root / "best_model.pth").exists())
            self.assertTrue((root / "loss_curve.png").exists())
            with (root / "valid_predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["dataset_id"] for r in rows], ["a", "b"])
            self.assertTrue(all("residual" in row for row in rows))
            self.assertFalse(any("test" in p.name for p in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
