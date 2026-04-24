from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import DVS128GestureFrames, PadCollate
from models.rsnn_02re import RSNNConfig, RSNN


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            a[k] = merge_dict(a[k], v)
        else:
            a[k] = v
    return a


def load_config(base_path):
    base = load_yaml(base_path)

    final_cfg = {}

    for item in base.get("defaults", []):
        for key, name in item.items():
            sub_path = Path(base_path).parent / key / f"{name}.yaml"
            sub_cfg = load_yaml(sub_path)
            final_cfg = merge_dict(final_cfg, sub_cfg)

    final_cfg = merge_dict(final_cfg, base)
    return final_cfg


def parse_value(raw: str):
    text = raw.strip()

    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None

    try:
        if text.startswith("0") and text not in ("0",) and not text.startswith(("0.", "0e", "0E")):
            raise ValueError
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, (list, dict, tuple, bool)) or parsed is None:
            return parsed
    except Exception:
        pass

    return text


def set_by_dotted_path(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg

    for i, k in enumerate(keys[:-1]):
        if k not in cur:
            raise KeyError(
                f"Unknown config path: '{dotted_key}' "
                f"(missing node: '{'.'.join(keys[:i+1])}')"
            )
        if not isinstance(cur[k], dict):
            raise KeyError(
                f"Config path '{'.'.join(keys[:i+1])}' is not a dict node, "
                f"cannot continue into '{dotted_key}'"
            )
        cur = cur[k]

    last = keys[-1]
    if last not in cur:
        raise KeyError(f"Unknown config key: '{dotted_key}'")

    old_value = cur[last]

    if old_value is not None:
        old_type = type(old_value)

        if isinstance(old_value, float) and isinstance(value, int):
            value = float(value)
        elif isinstance(old_value, bool) and isinstance(value, bool):
            pass
        elif isinstance(old_value, int) and isinstance(value, int) and not isinstance(value, bool):
            pass
        elif isinstance(old_value, float) and isinstance(value, float):
            pass
        elif isinstance(old_value, str) and isinstance(value, str):
            pass
        elif isinstance(old_value, list) and isinstance(value, list):
            pass
        elif isinstance(old_value, dict) and isinstance(value, dict):
            pass
        elif type(value) is not old_type:
            raise TypeError(
                f"Type mismatch for '{dotted_key}': "
                f"expected {old_type.__name__}, got {type(value).__name__}"
            )

    cur[last] = value


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected format: key=value"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key is not allowed.")

        value = parse_value(raw_value)
        set_by_dotted_path(cfg, key, value)

    return cfg


def seed_everything(seed: int = 42) -> None:
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(runtime_cfg: Dict[str, Any]) -> torch.device:
    requested = str(runtime_cfg.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sanitize_name(value: Any) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "unknown"


def infer_model_name(cfg: Dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    return sanitize_name(
        model_cfg.get("name")
        or model_cfg.get("type")
        or model_cfg.get("arch")
        or "rsnn"
    )


def infer_dataset_name(cfg: Dict[str, Any]) -> str:
    dcfg = cfg.get("dataset", {})
    return sanitize_name(
        dcfg.get("name")
        or dcfg.get("dataset_name")
        or Path(str(dcfg.get("root_dir", "dataset"))).name
        or "dataset"
    )


class BatchMetricLogger:
    def __init__(
        self,
        log_root: Path,
        cfg: Dict[str, Any],
        config_path: str,
        device: torch.device,
    ) -> None:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = start_time
        self.model_name = infer_model_name(cfg)
        self.dataset_name = infer_dataset_name(cfg)
        self.run_dir = log_root / self.model_name / self.dataset_name / start_time
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.yaml"
        self.meta_path = self.run_dir / "run_meta.json"

        cfg_to_dump = copy.deepcopy(cfg)
        cfg_to_dump.setdefault("experiment", {})
        cfg_to_dump["experiment"]["resolved_log_dir"] = str(self.run_dir)
        cfg_to_dump["experiment"]["run_start_time"] = start_time
        cfg_to_dump["experiment"]["config_path"] = str(config_path)
        cfg_to_dump["runtime"] = dict(cfg_to_dump.get("runtime", {}))
        cfg_to_dump["runtime"]["resolved_device"] = str(device)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_to_dump, f, allow_unicode=True, sort_keys=False)

        meta = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "start_time": start_time,
            "run_dir": str(self.run_dir),
            "metrics_file": str(self.metrics_path),
            "config_file": str(self.config_path),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def log(self, record: Dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_dataloaders(cfg: Dict[str, Any], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg["dataset"]
    lcfg = cfg["loader"]

    common_kwargs = dict(
        root=dcfg["root_dir"],
        window_us=dcfg["window_us"],
        spatial_size=dcfg["spatial_size"],
        max_steps=None if dcfg["max_steps"] in (-1, None) else dcfg["max_steps"],
        polarity_channels=dcfg["polarity_channels"],
        normalize=dcfg["normalize"],
    )

    train_ds = DVS128GestureFrames(train=True, **common_kwargs)
    test_ds = DVS128GestureFrames(train=False, **common_kwargs)

    collate_fn = PadCollate(pad_value=0.0)
    pin_memory = bool(lcfg["pin_memory"] and device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=lcfg["batch_size"],
        shuffle=True,
        num_workers=lcfg["num_workers"],
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=lcfg["batch_size"],
        shuffle=False,
        num_workers=lcfg["num_workers"],
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, test_loader


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = RSNNConfig(**cfg["model"])
    model = RSNN(model_cfg).to(device)
    return model


def build_criterion(train_cfg: Dict[str, Any]) -> nn.Module:
    name = str(train_cfg.get("criterion", "cross_entropy")).lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported criterion: {name}")


def maybe_apply_plasticity(model: nn.Module, state: Optional[Dict[str, torch.Tensor]]) -> None:
    if hasattr(model, "apply_plasticity"):
        model.apply_plasticity(state=state)
    elif hasattr(model, "apply_mechanisms"):
        model.apply_mechanisms(state=state)


def maybe_stabilize(model: nn.Module) -> None:
    if hasattr(model, "stabilize_"):
        model.stabilize_()
    elif hasattr(model, "core") and hasattr(model.core, "stabilize_"):
        model.core.stabilize_()


def collect_spiking_connectivity_stats(
    model: nn.Module,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    收集 Spiking / recurrent connectivity 相关统计。
    连接率优先使用 core.last_metrics["density"]，因为这是模型内部统一口径；
    state["rec_gate"] 只作为补充，用来记录 active connection 数等信息。
    """
    stats: Dict[str, float] = {}

    core = None
    if hasattr(model, "core"):
        core = model.core
    elif hasattr(model, "module") and hasattr(model.module, "core"):
        core = model.module.core

    if core is None:
        return stats

    # 1) 优先读模型内部已经记录好的 metrics
    last_metrics = getattr(core, "last_metrics", {}) or {}

    # 把 density 作为主连接率字段，保证 train_conn / test_conn 不会丢
    if "density" in last_metrics:
        stats["spike_conn_density"] = float(last_metrics["density"])
        stats["spike_conn_density_recorded"] = float(last_metrics["density"])

    for src_key, dst_key in [
        ("gate_soft_mean", "spike_conn_soft_density"),
        ("gate_score_mean", "spike_conn_gate_score_mean"),
        ("spike_rate", "spike_rate"),
        ("activity_mean", "spike_activity_mean"),
        ("activity_std", "spike_activity_std"),
        ("membrane_abs_mean", "spike_membrane_abs_mean"),
        ("rec_weight_abs_mean", "spike_rec_weight_abs_mean"),
        ("out_abs_mean", "spike_out_abs_mean"),
    ]:
        if src_key in last_metrics:
            stats[dst_key] = float(last_metrics[src_key])

    # 2) 如果 state 里有 rec_gate，再额外补充更细的 gate 统计
    if state is not None and isinstance(state, dict) and "rec_gate" in state:
        rec_gate = state["rec_gate"]
        valid_edges = core.valid_edges
        total_valid = float(valid_edges.sum().item())

        if total_valid > 0:
            stats["spike_conn_valid_total"] = total_valid

            if rec_gate.dim() == 3:
                active_per_sample = rec_gate.sum(dim=(1, 2)).float()
                stats["spike_conn_active_mean"] = float(active_per_sample.mean().item())
                stats["spike_conn_active_std"] = float(
                    active_per_sample.std(unbiased=False).item()
                )
                stats["spike_conn_density_from_gate"] = float(
                    (active_per_sample.mean() / total_valid).item()
                )
            elif rec_gate.dim() == 2:
                active = float(rec_gate.sum().item())
                stats["spike_conn_active"] = active
                stats["spike_conn_density_from_gate"] = active / total_valid

    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    include_aux_loss: bool = False,
    epoch: Optional[int] = None,
    metric_logger: Optional[BatchMetricLogger] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_correct = 0
    total_samples = 0
    spiking_sums: Dict[str, float] = {}

    batch_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    total_batches = len(data_loader)

    for batch_idx, batch in enumerate(batch_bar, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits, state = model(frames, lengths=lengths, mask=mask, return_state=True)
        main_loss = criterion(logits, labels)

        aux_loss = torch.zeros((), device=device, dtype=main_loss.dtype)
        if include_aux_loss and hasattr(model, "auxiliary_loss"):
            aux_loss = model.auxiliary_loss()

        loss = main_loss + aux_loss
        batch_correct = int((logits.argmax(dim=1) == labels).sum().item())
        spiking_stats = collect_spiking_connectivity_stats(model, state)

        bs = labels.size(0)
        total_main_loss += float(main_loss.item()) * bs
        total_aux_loss += float(aux_loss.item()) * bs
        total_loss += float(loss.item()) * bs
        total_correct += batch_correct
        total_samples += int(bs)

        for k, v in spiking_stats.items():
            spiking_sums[k] = spiking_sums.get(k, 0.0) + float(v) * bs

        postfix = {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
        }
        if "spike_conn_density" in spiking_stats:
            postfix["conn"] = spiking_stats["spike_conn_density"]
        if "spike_rate" in spiking_stats:
            postfix["spk"] = spiking_stats["spike_rate"]
        if include_aux_loss:
            postfix["main"] = total_main_loss / max(total_samples, 1)
            postfix["aux"] = total_aux_loss / max(total_samples, 1)
        batch_bar.set_postfix(postfix)

        if metric_logger is not None:
            metric_logger.log({
                "record_type": "batch",
                "split": "test",
                "epoch": epoch,
                "batch": batch_idx,
                "total_batches": total_batches,
                "batch_size": int(bs),
                "loss": float(loss.item()),
                "main_loss": float(main_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "acc": float(batch_correct / max(bs, 1)),
                **spiking_stats,
            })

    result = {
        "loss": total_loss / max(total_samples, 1),
        "main_loss": total_main_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }
    if include_aux_loss:
        result["aux_loss"] = total_aux_loss / max(total_samples, 1)

    for k, v in spiking_sums.items():
        result[k] = v / max(total_samples, 1)

    return result


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: Dict[str, Any],
    epoch: Optional[int] = None,
    metric_logger: Optional[BatchMetricLogger] = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_correct = 0
    total_samples = 0
    spiking_sums: Dict[str, float] = {}

    use_amp = bool(train_cfg["use_amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    batch_bar = tqdm(data_loader, desc="Training", leave=False)
    total_batches = len(data_loader)

    for batch_idx, batch in enumerate(batch_bar, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, state = model(frames, lengths=lengths, mask=mask, return_state=True)
            main_loss = criterion(logits, labels)
            aux_loss = model.auxiliary_loss() if hasattr(model, "auxiliary_loss") else torch.zeros((), device=device)
            loss = main_loss + float(train_cfg["aux_loss_weight"]) * aux_loss

        spiking_stats = collect_spiking_connectivity_stats(model, state)

        scaler.scale(loss).backward()

        grad_clip = train_cfg.get("grad_clip")
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if train_cfg.get("apply_plasticity_after_step", False):
            maybe_apply_plasticity(model, state)

        if train_cfg.get("stabilize_after_step", True):
            maybe_stabilize(model)

        batch_correct = int((logits.argmax(dim=1) == labels).sum().item())
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_main_loss += float(main_loss.item()) * bs
        total_aux_loss += float(aux_loss.item()) * bs
        total_correct += batch_correct
        total_samples += int(bs)

        for k, v in spiking_stats.items():
            spiking_sums[k] = spiking_sums.get(k, 0.0) + float(v) * bs

        postfix = {
            "loss": total_loss / max(total_samples, 1),
            "main": total_main_loss / max(total_samples, 1),
            "aux": total_aux_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
        }
        if "spike_conn_density" in spiking_stats:
            postfix["conn"] = spiking_stats["spike_conn_density"]
        if "spike_rate" in spiking_stats:
            postfix["spk"] = spiking_stats["spike_rate"]
        batch_bar.set_postfix(postfix)

        if metric_logger is not None:
            metric_logger.log({
                "record_type": "batch",
                "split": "train",
                "epoch": epoch,
                "batch": batch_idx,
                "total_batches": total_batches,
                "batch_size": int(bs),
                "loss": float(loss.item()),
                "main_loss": float(main_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "acc": float(batch_correct / max(bs, 1)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **spiking_stats,
            })

    result = {
        "loss": total_loss / max(total_samples, 1),
        "main_loss": total_main_loss / max(total_samples, 1),
        "aux_loss": total_aux_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }
    for k, v in spiking_sums.items():
        result[k] = v / max(total_samples, 1)

    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epoch: int,
    best_acc: float,
) -> None:
    model_cfg = model.cfg if hasattr(model, "cfg") else None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_cfg": asdict(model_cfg) if model_cfg is not None else None,
            "full_config": cfg,
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def main(config_path: str, overrides: Optional[list[str]] = None) -> None:
    cfg = load_config(config_path)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    seed_everything(cfg["experiment"]["seed"])
    device = resolve_device(cfg["runtime"])
    save_dir = Path(cfg["experiment"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    log_root = Path(cfg["experiment"].get("logs_dir", save_dir / "logs"))
    metric_logger = BatchMetricLogger(
        log_root=log_root,
        cfg=cfg,
        config_path=config_path,
        device=device,
    )

    train_loader, test_loader = build_dataloaders(cfg, device)
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = build_criterion(cfg["train"])

    tqdm.write(f"device: {device}")
    tqdm.write(f"dataset config: {cfg['dataset']}")
    tqdm.write(f"loader config: {cfg['loader']}")
    tqdm.write(f"model config: {asdict(model.cfg)}")
    tqdm.write(f"train batches: {len(train_loader)} | test batches: {len(test_loader)}")
    tqdm.write(f"log dir: {metric_logger.run_dir}")

    best_acc = -1.0
    best_path = save_dir / "best_model.pt"

    epoch_bar = tqdm(range(1, cfg["train"]["epochs"] + 1), desc="Epochs")
    for epoch in epoch_bar:
        start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_cfg=cfg["train"],
            epoch=epoch,
            metric_logger=metric_logger,
        )
        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            include_aux_loss=cfg["train"].get("include_aux_loss_in_eval", False),
            epoch=epoch,
            metric_logger=metric_logger,
        )

        elapsed = time.time() - start
        tqdm.write(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_conn={train_metrics.get('spike_conn_density', float('nan')):.4f} "
            f"train_spk={train_metrics.get('spike_rate', float('nan')):.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_conn={test_metrics.get('spike_conn_density', float('nan')):.4f} "
            f"test_spk={test_metrics.get('spike_rate', float('nan')):.4f} "
            f"time={elapsed:.2f}s"
        )

        metric_logger.log({
            "record_type": "epoch",
            "epoch": epoch,
            "elapsed_sec": float(elapsed),
            "train": train_metrics,
            "test": test_metrics,
            "best_acc_before_update": float(best_acc),
        })

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            save_checkpoint(best_path, model, optimizer, cfg, epoch, best_acc)
            metric_logger.log({
                "record_type": "checkpoint",
                "epoch": epoch,
                "best_acc": float(best_acc),
                "checkpoint_path": str(best_path),
            })

    tqdm.write(f"best_acc={best_acc:.4f}")
    tqdm.write(f"best_model_saved_to={best_path}")
    tqdm.write(f"metrics_saved_to={metric_logger.metrics_path}")
    tqdm.write(f"config_saved_to={metric_logger.config_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train refactored RSNN on DVS128 Gesture")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/rsnn_dvs128gesture.default.yaml",
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help=(
            "Override config values with dotted paths, "
            "e.g. model.use_sparse_connectivity=true train.lr=1e-4"
        ),
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args.config, args.set)
