import os
import random
import torch
import yaml
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from . import readers
from .datasets import CSIDataset
from .utils import load_params, train_model, resize_csi_to_fixed_length, load_custom_model
from .utils.cache import get_cache_key, load_cache, save_cache
from .processors.base_processor import BaseProcessor
from .models import CSIModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Tuple, Callable

import logging

logger = logging.getLogger(__name__)


def _raise_no_usable_samples(input_path: str, dataset: str) -> None:
    raise ValueError(
        f"No usable samples were produced from '{input_path}' for dataset '{dataset}'. "
        "All files may have been filtered out during reading/preprocessing or had invalid filenames."
    )


def _load_and_preprocess(
    input_path: str,
    dataset: str,
    pad_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load CSI data, run processing pipeline, and return arrays ready for splitting.

    Returns:
        (processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels)
    """
    csi_data_list = readers.load_data(input_path, dataset)

    processor = BaseProcessor()
    res = processor.process(csi_data_list, dataset=dataset)

    unadjusted_data = res[0]
    if not unadjusted_data:
        _raise_no_usable_samples(input_path, dataset)

    processed_data = resize_csi_to_fixed_length(unadjusted_data, target_length=pad_len)
    if not processed_data:
        _raise_no_usable_samples(input_path, dataset)

    logger.info(f"processed_data's shape: {processed_data[0].shape}")

    labels = res[1]
    groups = res[2]

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = [label_map[label] for label in labels]

    unique_groups = sorted(list(set(groups)))
    group_map = {group: i for i, group in enumerate(unique_groups)}
    zero_indexed_groups = [group_map[group] for group in groups]

    logger.info(f"all unique labels idx: {list(set(zero_indexed_labels))}")
    logger.info(f"all unique groups idx: {list(set(zero_indexed_groups))}")
    logger.info(f"total sample: {len(processed_data)}, "
                f"total labels: {len(zero_indexed_labels)}, total groups: {len(zero_indexed_groups)}")

    processed_data = np.array(processed_data)
    zero_indexed_labels = np.array(zero_indexed_labels)
    zero_indexed_groups = np.array(zero_indexed_groups)

    return processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels


def _create_data_split(
    processed_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_split: float,
    val_split: float,
    seed: int,
    use_simple_split: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets.

    Returns:
        (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    try:
        if use_simple_split:
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                processed_data, labels,
                test_size=test_split, random_state=seed
            )
            test_data, val_data, test_labels, val_labels = train_test_split(
                temp_data, temp_labels,
                test_size=val_split, random_state=seed
            )
        else:
            splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
            train_idx, temp_idx = next(
                splitter_1.split(processed_data, labels, groups=groups)
            )

            train_data = processed_data[train_idx]
            train_labels = labels[train_idx]

            temp_data = processed_data[temp_idx]
            temp_labels = labels[temp_idx]
            temp_groups = groups[temp_idx]

            splitter_2 = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
            test_idx, val_idx = next(splitter_2.split(temp_data, temp_labels, groups=temp_groups))

            test_data = temp_data[test_idx]
            test_labels = temp_labels[test_idx]

            val_data = temp_data[val_idx]
            val_labels = temp_labels[val_idx]
    except ValueError as exc:
        raise ValueError(
            f"Unable to create train/val/test splits from {len(processed_data)} usable sample(s). "
            "Please provide more data or adjust test_split/val_split."
        ) from exc

    split_sizes = {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data),
    }
    empty_splits = [name for name, size in split_sizes.items() if size == 0]
    if empty_splits:
        raise ValueError(
            f"Unable to create non-empty train/val/test splits from {len(processed_data)} usable sample(s). "
            f"Empty split(s): {', '.join(empty_splits)}. Please provide more data or adjust test_split/val_split."
        )

    train_data = np.stack(train_data, axis=0)
    val_data = np.stack(val_data, axis=0)
    test_data = np.stack(test_data, axis=0)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[list, list, float]:
    """Evaluate model on test set.

    Returns:
        (predictions, labels, accuracy)
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (csi_data_batch, test_labels_batch) in enumerate(
            tqdm(test_loader, desc="Evaluating", leave=False)
        ):
            csi_data_batch = csi_data_batch.to(device)
            test_labels_batch = test_labels_batch.to(device)

            outputs = model(csi_data_batch)

            _, predicted_classes = torch.max(outputs.data, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(test_labels_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return all_predictions, all_labels, accuracy


def pipeline(
    input_path: str,
    output_folder: str,
    dataset: str,
    model_path: Optional[str] = None,
    # Hyperparameter overrides
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    num_epochs: Optional[int] = None,
    padding_length: Optional[int] = None,
    test_split: float = 0.3,
    val_split: float = 0.5,
    num_seeds: int = 5,
    config_file: Optional[str] = None,
    num_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_cache: bool = True,
) -> None:
    """
    Run the full CSI classification pipeline.

    Args:
        input_path: Path to input data directory
        output_folder: Path to output directory
        dataset: Dataset name
        model_path: Optional path to custom model file
        batch_size: Override default batch size
        learning_rate: Override default learning rate
        weight_decay: Override default weight decay
        num_epochs: Override default number of epochs
        padding_length: Override default padding length
        test_split: Fraction of data held out (not used for training) (default 0.3)
        val_split: Fraction of held-out data used for validation (default 0.5)
        num_seeds: Number of random seeds to run (default 5)
        config_file: Optional YAML config file to load parameters from
        num_workers: Number of DataLoader workers. When None, auto-detects: min(cpu_count, 8)
        progress_callback: Optional callable invoked after each training epoch with a dict of metrics
        use_cache: If True, cache preprocessed data to avoid re-processing on repeated runs (default True)
    """
    ipath = input_path
    os.makedirs(output_folder, exist_ok=True)
    opath = Path(output_folder)
    dataset_name = dataset

    # Resolve num_workers
    effective_num_workers = num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)

    if model_path is not None:
        logger.info(f"Loading model from {model_path}")
    else:
        logger.info("Loading default model")

    # Load default params
    try:
        params = load_params(dataset_name)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"{e}")
        return

    # Load YAML config overrides if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            yaml_params = yaml.safe_load(f)
        if yaml_params and dataset_name in yaml_params:
            params.update(yaml_params[dataset_name])
        logger.info(f"Loaded config from {config_file}")

    # Apply function-level overrides (highest priority)
    batch = batch_size if batch_size is not None else params.get("batch", 32)
    lr = learning_rate if learning_rate is not None else params.get("lr", 3e-4)
    wd = weight_decay if weight_decay is not None else params.get("wd", 1e-3)
    num_epochs_val = num_epochs if num_epochs is not None else params.get("num_epochs", 20)
    pad_len = padding_length if padding_length is not None else params.get("padding_length", 1500)

    random_seeds = [random.randint(0, 999) for _ in range(num_seeds)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Hyperparameters: batch={batch}, lr={lr}, wd={wd}, epochs={num_epochs_val}, pad={pad_len}")

    # begin to preprocess, training and eval
    cache_dir = os.path.join(output_folder, '.wsdp_cache')
    cached_result = None
    cache_key = None
    if use_cache:
        cache_key = get_cache_key(ipath, dataset_name, pad_len)
        cached_result = load_cache(cache_dir, cache_key)

    if cached_result is not None:
        logger.info("Cache hit: loaded preprocessed data from cache")
        processed_data = cached_result['processed_data']
        zero_indexed_labels = cached_result['labels']
        zero_indexed_groups = cached_result['groups']
        unique_labels = cached_result['unique_labels']
    else:
        if use_cache:
            logger.info("Cache miss: processing data from scratch")
        processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels = \
            _load_and_preprocess(ipath, dataset_name, pad_len)
        if use_cache and cache_key is not None:
            save_cache(cache_dir, cache_key, processed_data, zero_indexed_labels,
                       zero_indexed_groups, unique_labels)

    logger.info(f"the following {num_seeds} seeds will be used: {random_seeds}")

    top1_accuracies = []

    # Check if we have enough groups for GroupShuffleSplit
    n_groups = len(set(zero_indexed_groups))
    use_simple_split = n_groups < 3
    if use_simple_split:
        logger.warning(f"Only {n_groups} group(s) found (< 3). "
                       f"Using simple train_test_split instead of GroupShuffleSplit.")

    for i, current_seed in enumerate(random_seeds):
        print(f"\n{'=' * 25} epoch {i + 1}/{len(random_seeds)} "
              f"begin (Random State: {current_seed}) {'=' * 25}\n")

        train_data, val_data, test_data, train_labels, val_labels, test_labels = \
            _create_data_split(
                processed_data, zero_indexed_labels, zero_indexed_groups,
                test_split, val_split, current_seed, use_simple_split,
            )

        logger.info(f"num of samples in train_data: {len(train_data)}, "
                     f"num of samples in test_data: {len(test_data)}, num of samples in val_data: {len(val_data)}")
        logger.info(f"shape of first sample of train_data: {train_data[0].shape}, "
                     f"shape of last sample of train_data: {train_data[-1].shape}")

        train_dataset = CSIDataset(train_data, train_labels)
        test_dataset = CSIDataset(test_data, test_labels)
        val_dataset = CSIDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=False)

        num_classes = len(unique_labels)
        if model_path is None:
            model = CSIModel(num_classes=num_classes)
        else:
            model = load_custom_model(model_path, num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        checkpoint_path = opath / f"best_checkpoint_{current_seed}.pth"

        logger.info("begin training")
        training_history = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs_val,
            device=device,
            checkpoint_path=checkpoint_path,
            padding_length=pad_len,
            progress_callback=progress_callback,
        )

        logger.info(f"training complete, save training_history to: "
                    f"{opath / ('training_history_' + str(current_seed) + '.csv')}")

        df = pd.DataFrame(training_history)
        df.to_csv(opath / f"training_history_{current_seed}.csv", index_label='epoch')

        logger.info("save successfully, begin to evaluate model")
        cp = checkpoint_path
        if not os.path.isfile(cp):
            raise FileNotFoundError(f" no model in file path: {cp}")

        logger.info(f"loading model from {cp} ...")
        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        all_predictions, all_labels, current_top1_acc = _evaluate_model(model, test_loader, device)

        logger.info("eval complete")

        top1_accuracies.append(current_top1_acc)
        logger.info(f"Top-1 acc of current epoch: {current_top1_acc:.4f}")

        logger.info("classification report:\n" + classification_report(all_labels, all_predictions))

        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Random State: {current_seed})", fontsize=16)
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()

        figure_path = opath / f"cm_rs_{current_seed}.png"
        plt.savefig(figure_path)
        plt.close()

    accuracies_np = np.array(top1_accuracies)
    mean_accuracy = np.mean(accuracies_np)
    variance_accuracy = np.var(accuracies_np)

    logger.info(f"All {len(random_seeds)} Top-1 acc: {[f'{acc:.4f}' for acc in top1_accuracies]}")
    logger.info(f"Avg Top-1 acc: {mean_accuracy:.4f}")
    logger.info(f"Variance of Top-1 acc: {variance_accuracy:.6f}")
    logger.info("All pipeline complete")
