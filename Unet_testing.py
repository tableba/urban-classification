from copy import deepcopy
from time import time
from src.dataset.read_data import ReadTifs
from src.dataset.SLIC import preprocess_homogenise, preprocess_extra_channel, postprocess_consistency
import torch
from torch.utils.data import DataLoader
from src.dataset.preprocessing import normalize_bands, add_spectral_indices, extract_patches, SentinelDataset
from src.models.Unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRATEGIES = ["baseline", "homogenise", "extra_channel", "postprocess"]
DEFAULT_SEEDS = [42, 123, 456]
CHECKPOINT_DIR = "checkpoints"
CLASS_NAMES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker so augmentation is reproducible across workers."""
    base_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def prepare_data(max_tiles_per_third=10, seed=42):
    all_patches = []
    reader = ReadTifs()

    s2_files = {
        os.path.splitext(f)[0].split("_")[1]: os.path.join(reader.s2_dir, f)
        for f in os.listdir(reader.s2_dir)
        if f.lower().endswith(".tif")
    }
    dw_files = {
        os.path.splitext(f)[0].split("_")[1]: os.path.join(reader.dw_dir, f)
        for f in os.listdir(reader.dw_dir)
        if f.lower().endswith(".tif")
    }
    common_tiles = sorted(set(s2_files.keys()) & set(dw_files.keys()))

    third = len(common_tiles) // 3
    first_third = common_tiles[:third]
    second_third = common_tiles[third:2*third]
    last_third = common_tiles[2*third:]

    rng = random.Random(seed)
    selected_tiles = (
        rng.sample(first_third, min(max_tiles_per_third, len(first_third))) +
        rng.sample(second_third, min(max_tiles_per_third, len(second_third))) +
        rng.sample(last_third, min(max_tiles_per_third, len(last_third)))
    )

    print(f"Selected tiles: {selected_tiles}")

    tiles_processed = 0
    for key in selected_tiles:
        print(f"Processing tile: {key}")
        s2_path = s2_files[key]
        dw_path = dw_files[key]
        s2_data = reader._read_tif(s2_path)
        dw_data = reader._read_tif(dw_path)

        for s2_patch, dw_patch in extract_patches(s2_data, dw_data):
            all_patches.append((key, s2_patch, dw_patch))

        tiles_processed += 1
        print(f"Patches extracted from tile {key}: {sum(1 for patch in all_patches if patch[0] == key)} (Total tiles processed: {tiles_processed}/{len(selected_tiles)})")

    return all_patches


def split_tiles(patches, test_ratio=0.1, val_ratio=0.1, seed=42):
    tile_map = {}
    for tile_key, s2_patch, dw_patch in patches:
        tile_map.setdefault(tile_key, []).append((s2_patch, dw_patch))

    tile_ids = list(tile_map.keys())
    rng = random.Random(seed)
    rng.shuffle(tile_ids)

    n_tiles = len(tile_ids)
    n_test = max(1, min(n_tiles - 2, int(n_tiles * test_ratio)))
    n_val = max(1, min(n_tiles - 1 - n_test, int(n_tiles * val_ratio)))
    n_train = n_tiles - n_test - n_val

    train_tiles = tile_ids[:n_train]
    val_tiles = tile_ids[n_train:n_train + n_val]
    test_tiles = tile_ids[n_train + n_val:]

    train = [(tile, s2, dw) for tile in train_tiles for s2, dw in tile_map[tile]]
    val = [(tile, s2, dw) for tile in val_tiles for s2, dw in tile_map[tile]]
    test = [(tile, s2, dw) for tile in test_tiles for s2, dw in tile_map[tile]]

    print(f"Split tiles -> train: {len(train_tiles)}, val: {len(val_tiles)}, test: {len(test_tiles)}")
    return train, val, test


def drop_tile_keys(patches):
    return [(s2, dw) for _, s2, dw in patches]


def transform_patch(s2_patch, strategy):
    # extract_patches already runs clean_s2_data; only normalisation is needed here.
    s2 = normalize_bands(s2_patch)

    if strategy in ["baseline", "postprocess"]:
        return add_spectral_indices(s2)

    if strategy == "homogenise":
        s2_tensor = torch.from_numpy(s2.astype(np.float32)).unsqueeze(0)
        s2_hom = preprocess_homogenise(s2_tensor).squeeze(0).numpy()
        return add_spectral_indices(s2_hom)

    if strategy == "extra_channel":
        s2_base = add_spectral_indices(s2)
        s2_extra = preprocess_extra_channel(torch.from_numpy(s2_base.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
        return s2_extra

    raise ValueError(f"Unknown strategy: {strategy}")


def build_dataset(patches, strategy, augment=False):
    transformed = []
    for _, s2_patch, dw_patch in patches:
        transformed.append((transform_patch(s2_patch, strategy), dw_patch))
    return SentinelDataset(transformed, augment=augment)


def init_model(strategy, load_path=None, class_weights=None):
    print(f"Initializing model for strategy: {strategy}")
    in_channels = 13 if strategy == "extra_channel" else 12
    model = UNet(in_channels=in_channels).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        print(f"Model loaded from {load_path}")
    else:
        print("Training model from scratch.")

    return model, criterion, optimizer


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, strategy, patience=10):
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    wait = patience

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (s2_batch, dw_batch) in enumerate(train_loader):
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
            outputs = model(s2_batch)
            loss_ce = criterion(outputs, dw_batch)
            loss_dice = dice_loss(outputs, dw_batch)
            loss = loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        val_loss = evaluate_model(model, val_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            wait = patience
            print(f"New best validation loss: {best_loss:.4f}")
        else:
            wait -= 1
            print(f"No improvement. Patience left: {wait}")
            if wait <= 0:
                print("Early stopping!")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, dataloader, criterion):
    print("Evaluating model...")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (s2_batch, dw_batch) in enumerate(dataloader):
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
            outputs = model(s2_batch)
            loss_ce = criterion(outputs, dw_batch)
            loss_dice = dice_loss(outputs, dw_batch)
            total_loss += (loss_ce + loss_dice).item()
            if i % 10 == 9:
                print(f"Batch [{i+1}/{len(dataloader)}], Loss: {(loss_ce + loss_dice).item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def predict(model, dataloader, strategy):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for s2_batch, dw_batch in dataloader:
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
            logits = model(s2_batch)
            if strategy == "postprocess":
                logits = postprocess_consistency(logits, s2_batch)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            labels.append(dw_batch.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def boundary_map(label):
    boundary = np.zeros_like(label, dtype=bool)
    boundary[:-1, :] |= label[:-1, :] != label[1:, :]
    boundary[1:, :] |= label[:-1, :] != label[1:, :]
    boundary[:, :-1] |= label[:, :-1] != label[:, 1:]
    boundary[:, 1:] |= label[:, :-1] != label[:, 1:]
    return boundary


def compute_segmentation_metrics(predictions, targets, num_classes=9, eps=1e-6):
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()
    mask = (tgt_flat >= 0) & (tgt_flat < num_classes)
    idx = tgt_flat[mask] * num_classes + pred_flat[mask]
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape((num_classes, num_classes))

    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1).astype(np.float64) - tp
    fp = cm.sum(axis=0).astype(np.float64) - tp
    union = tp + fp + fn

    iou = tp / (union + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    support = cm.sum(axis=1)
    valid = support > 0
    miou = np.nanmean(np.where(valid, iou, np.nan))
    precision_macro = np.nanmean(np.where(valid, precision, np.nan))
    recall_macro = np.nanmean(np.where(valid, recall, np.nan))
    f1_macro = np.nanmean(np.where(valid, f1, np.nan))

    pixel_accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0

    bf_scores = []
    for pred, target in zip(predictions, targets):
        pred_b = boundary_map(pred)
        tgt_b = boundary_map(target)
        tp_b = np.logical_and(pred_b, tgt_b).sum()
        fp_b = pred_b.sum() - tp_b
        fn_b = tgt_b.sum() - tp_b
        if pred_b.sum() == 0 and tgt_b.sum() == 0:
            bf = 1.0
        else:
            bf = 2 * tp_b / (2 * tp_b + fp_b + fn_b + eps)
        bf_scores.append(bf)

    return {
        "per_class_iou": iou.tolist(),
        "mIoU": float(miou),
        "pixel_accuracy": float(pixel_accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "bf_score": float(np.mean(bf_scores)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_metrics(model, dataloader, strategy):
    preds, labels = predict(model, dataloader, strategy)
    return compute_segmentation_metrics(preds, labels)


def compute_class_weights(patches, num_classes=9, eps=1e-6):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, dw in patches:
        labels = dw.squeeze() if dw.ndim == 3 and dw.shape[0] == 1 else dw
        counts += np.bincount(labels.flatten(), minlength=num_classes)

    freq = counts / counts.sum()
    weights = 1.0 / (freq + eps)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def dice_loss(outputs, targets, num_classes=9, eps=1e-6):
    probs = torch.softmax(outputs, dim=1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
    cardinality = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_score.mean()


def build_dataloaders(train_patches, val_patches, test_patches, strategy):
    train_dataset = build_dataset(train_patches, strategy, augment=True)
    val_dataset = build_dataset(val_patches, strategy, augment=False)
    test_dataset = build_dataset(test_patches, strategy, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=_worker_init_fn, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=_worker_init_fn, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=_worker_init_fn, persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, test_loader


def run_strategy(strategy, train_patches, val_patches, test_patches, seed, checkpoint_dir=CHECKPOINT_DIR):
    class_weights = compute_class_weights(drop_tile_keys(train_patches), num_classes=9).to(DEVICE)
    train_loader, val_loader, test_loader = build_dataloaders(train_patches, val_patches, test_patches, strategy)
    model, criterion, optimizer = init_model(strategy, class_weights=class_weights)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, strategy)
    metrics = evaluate_metrics(model, test_loader, strategy)
    print(f"Strategy {strategy} completed. Test mIoU={metrics['mIoU']:.4f}, pixel_acc={metrics['pixel_accuracy']:.4f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"unet_{strategy}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")

    metrics_path = os.path.join(checkpoint_dir, f"metrics_{strategy}_seed{seed}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


def run_experiments(patches, strategies=STRATEGIES, seeds=DEFAULT_SEEDS, checkpoint_dir=CHECKPOINT_DIR):
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {strategy: [] for strategy in strategies}
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        set_seed(seed)
        train_patches, val_patches, test_patches = split_tiles(patches, seed=seed)

        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            metrics = run_strategy(strategy, train_patches, val_patches, test_patches, seed, checkpoint_dir=checkpoint_dir)
            results[strategy].append(metrics)

            # Persist running results so a mid-experiment crash doesn't lose everything.
            results_path = os.path.join(checkpoint_dir, "all_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    # Aggregate scalar metrics with mean/std; aggregate per-class IoU per-class;
    # average confusion matrices element-wise (do not take std).
    SCALAR_METRICS = ["mIoU", "pixel_accuracy", "precision", "recall", "f1", "bf_score"]

    summary = {}
    for strategy, metrics_list in results.items():
        summary[strategy] = {}
        for metric in SCALAR_METRICS:
            values = np.array([m[metric] for m in metrics_list], dtype=np.float64)
            summary[strategy][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }

        per_class_arr = np.array([m["per_class_iou"] for m in metrics_list], dtype=np.float64)
        summary[strategy]["per_class_iou"] = {
            "mean": per_class_arr.mean(axis=0).tolist(),
            "std": per_class_arr.std(axis=0, ddof=0).tolist(),
        }

        cm_arr = np.array([m["confusion_matrix"] for m in metrics_list], dtype=np.float64)
        summary[strategy]["confusion_matrix"] = {
            "mean": cm_arr.mean(axis=0).tolist(),
            "sum": cm_arr.sum(axis=0).tolist(),
        }

    summary["_meta"] = {"class_names": CLASS_NAMES}

    summary_path = os.path.join(checkpoint_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregated summary to {summary_path}")

    return summary


def format_summary(summary):
    formatted = []
    for strategy, metrics in summary.items():
        if strategy.startswith("_"):
            continue
        formatted.append(f"Strategy: {strategy}")
        for metric, stats in metrics.items():
            if metric == "per_class_iou":
                means = stats["mean"]
                stds = stats["std"]
                pieces = ", ".join(
                    f"{name}={m:.4f}±{s:.4f}"
                    for name, m, s in zip(CLASS_NAMES, means, stds)
                )
                formatted.append(f"  per_class_iou: [{pieces}]")
            elif metric == "confusion_matrix":
                formatted.append(f"  confusion_matrix (mean over seeds): see summary.json")
            else:
                formatted.append(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    return "\n".join(formatted)


def visualize_predictions(model, dataloader, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, (s2_batch, dw_batch) in enumerate(dataloader):
            if i >= num_samples:
                break
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
            outputs = model(s2_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = dw_batch.cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(labels[0], cmap="jet")
            plt.title("Ground Truth")
            plt.subplot(1, 2, 2)
            plt.imshow(preds[0], cmap="jet")
            plt.title("Prediction")
            plt.show()


def print_class_distribution(patches):
    labels = []
    for _, dw in patches:
        labels.extend(dw.flatten())
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} pixels")


if __name__ == "__main__":
    print("=== Starting Unet Experiment Runner ===")

    start_time = time()
    patches = prepare_data(seed=42)
    summary = run_experiments(patches, strategies=["baseline"], seeds=[42])
    end_time = time()
    elapsed = end_time - start_time
    print("\n=== Experiment Summary ===")
    print(format_summary(summary))
    print(f"\nTotal execution time: {elapsed:.2f} seconds")