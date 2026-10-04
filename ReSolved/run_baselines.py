"""Run full random then scaffold training using the active Python environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    """顺序运行两套固定划分；失败即停止，不覆盖任何已有实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results_d3tales" / "local_full_baseline")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bond-features", choices=["legacy", "conjugation"], default="legacy")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or creating files")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    commands = []
    for split in ("random", "scaffold"):
        # 在启动第一套训练之前检查两套目录，避免训练到一半才发现冲突。
        output = args.output_root.resolve() / f"{split}_seed42_{args.epochs}epochs"
        # Check BOTH outputs before launching either job. Never delete old results.
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            parser.error(f"Output already exists and cannot be overwritten: {output}")
        commands.append([
            sys.executable, "-u", str(ROOT / "run_d3tales.py"),
            "--mode", "train", "--split", split, "--device", args.device,
            "--epochs", str(args.epochs), "--seed", "42", "--batch-size", "32",
            "--lr", "1e-4", "--weight-decay", "1e-5", "--output", str(output),
            "--bond-features", args.bond_features,
        ])
    print("Full Train/Validation training; Test remains SEALED.", flush=True)
    for index, command in enumerate(commands, 1):
        # 使用当前解释器维持已激活环境；dry-run 仅展示命令，不创建结果。
        print(f"[{index}/2] " + " ".join(command), flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Training failed (exit {exc.returncode}); sequence stopped.", file=sys.stderr)
                return exc.returncode
            except KeyboardInterrupt:
                print("Interrupted; sequence stopped.", file=sys.stderr)
                return 130
    if not args.dry_run:
        print("Both full training runs completed. Test remains SEALED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
