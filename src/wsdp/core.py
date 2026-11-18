import multiprocessing as mp
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
from .dataset_policy import uses_phase_amplitude
from .datasets import CSIDataset
from .structure import ChunkedCSIArray
from .utils import load_params, train_model, load_custom_model
from .utils.cache import get_cache_key, load_cache, save_cache
from .processors.base_processor import BaseProcessor
from .processors import ConfigurableProcessor
from .models import create_model
from .algorithms import apply_preset, load_config as load_algorithm_config
from .record import SeedRecord, persist_pipeline_record
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, Optional, Tuple, Callable

import logging

logger = logging.getLogger(__name__)


def _load_and_preprocess(
    input_path: str,
    dataset: str,
    pad_len: int,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
    reader: Optional[str] = None,
    storage_path: Optional[str] = None,
    storage_mode: str = "memory",
    storage_dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load CSI data, run processing pipeline, and return arrays ready for splitting.

    Returns:
        (processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels).
        For ``xrf55`` the groups are the raw filename trial ids (1-20), kept
        unmodified because ``_create_xrf55_repetition_split`` keys on them.
    """
    reader_name = reader or dataset
    if pipeline_steps is None:
        processor = BaseProcessor()
    else:
        processor = ConfigurableProcessor(pipeline_steps)

    # Direct callers get the same bounded implementation; no storage path is
    # needed because the default result lives in shared memory, not a file.
    from .processors.streaming import preprocess_to_memmap

    processed_data, labels, groups = preprocess_to_memmap(
        input_path,
        dataset,
        pad_len,
        processor,
        reader_name,
        storage_path,
        storage_mode=storage_mode,
        storage_dtype=storage_dtype,
    )

    logger.info(f"processed_data's shape: {processed_data.shape}")

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = [label_map[label] for label in labels]

    if dataset == 'xrf55':
        # xrf55 groups are filename trial ids (1-20); re-indexing them would
        # break the repetition split whenever a subset of trials is present.
        zero_indexed_groups = list(groups)
    else:
        unique_groups = sorted(list(set(groups)))
        group_map = {group: i for i, group in enumerate(unique_groups)}
        zero_indexed_groups = [group_map[group] for group in groups]

    logger.info(f"all unique labels idx: {list(set(zero_indexed_labels))}")
    logger.info(f"all unique groups idx: {list(set(zero_indexed_groups))}")
    logger.info(f"total sample: {len(processed_data)}, "
                f"total labels: {len(zero_indexed_labels)}, total groups: {len(zero_indexed_groups)}")

    zero_indexed_labels = np.array(zero_indexed_labels)
    zero_indexed_groups = np.array(zero_indexed_groups)

    return processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels


def _resolve_pipeline_steps(
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
    algorithm_config_file: Optional[str] = None,
    algorithm_preset: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Resolve optional algorithm pipeline configuration.

    Priority: explicit pipeline_steps > config file > preset > legacy BaseProcessor.
    Returning None preserves the historical BaseProcessor behavior.
    """
    if pipeline_steps is not None:
        return pipeline_steps
    if algorithm_config_file is not None:
        return load_algorithm_config(algorithm_config_file)
    if algorithm_preset is not None:
        return apply_preset(algorithm_preset)
    return None


def _create_data_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    test_split: float,
    val_split: float,
    seed: int,
    use_simple_split: bool,
    dataset: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Create split indices without copying any CSI sample data.

    The fourth return value reports whether the XRF55 repetition protocol was
    used and therefore whether its train-only normalization should be applied.
    """
    if dataset == "xrf55":
        try:
            train_idx, val_idx, test_idx = _create_xrf55_repetition_indices(groups)
            return train_idx, val_idx, test_idx, True
        except RuntimeError:
            # Trial subsets may lack the fixed protocol ranges (13-16 val,
            # 17-20 test). Fall back to a repetition-disjoint group split,
            # which preserves the protocol's cross-repetition intent.
            logger.warning(
                "XRF55 trial subset misses a protocol range; falling back to "
                "repetition-disjoint GroupShuffleSplit. Train-only statistics "
                "for configured normalization are skipped in this mode."
            )

    if use_simple_split:
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices, test_size=test_split, random_state=seed
        )
        test_idx, val_idx = train_test_split(
            temp_idx, test_size=val_split, random_state=seed
        )
    else:
        indices = np.arange(len(labels))
        splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
        train_idx, temp_idx = next(
            splitter_1.split(indices, labels, groups=groups)
        )

        temp_labels = labels[temp_idx]
        temp_groups = groups[temp_idx]
        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        test_local_idx, val_local_idx = next(
            splitter_2.split(temp_idx, temp_labels, groups=temp_groups)
        )
        test_idx = temp_idx[test_local_idx]
        val_idx = temp_idx[val_local_idx]

    return train_idx, val_idx, test_idx, False


def _create_data_split(
    processed_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_split: float,
    val_split: float,
    seed: int,
    use_simple_split: bool,
    dataset: Optional[str] = None,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a data split for backward-compatible external callers."""
    train_idx, val_idx, test_idx, xrf55_protocol = _create_data_split_indices(
        labels, groups, test_split, val_split, seed, use_simple_split, dataset=dataset
    )
    normalization = None
    if xrf55_protocol:
        normalization = _fit_xrf55_normalization(
            processed_data, train_idx, pipeline_steps
        )
    return _materialize_data_split(
        processed_data,
        labels,
        train_idx,
        val_idx,
        test_idx,
        normalization,
    )


def _create_xrf55_repetition_indices(
    repetitions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices for XRF55 trial ids 1-20."""
    repetitions = np.asarray(repetitions).astype(int)

    # In older preprocessing flows groups may have been zero-indexed to 0-19.
    # Convert that representation back to the filename's 1-20 trial ids.
    unique_repetitions = sorted(set(repetitions.tolist()))
    if unique_repetitions and unique_repetitions[0] == 0 and unique_repetitions[-1] == 19:
        repetitions = repetitions + 1

    invalid_mask = (repetitions < 1) | (repetitions > 20)
    if np.any(invalid_mask):
        invalid_values = sorted(set(repetitions[invalid_mask].tolist()))
        raise ValueError(f"Unexpected XRF55 repetition ids: {invalid_values}")

    train_mask = (repetitions >= 1) & (repetitions <= 12)
    val_mask = (repetitions >= 13) & (repetitions <= 16)
    test_mask = (repetitions >= 17) & (repetitions <= 20)

    if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
        raise RuntimeError(
            "XRF55 repetition split produced an empty split. "
            "Expected trial ids 1-20 from user_action_trial filenames."
        )

    return (
        np.flatnonzero(train_mask),
        np.flatnonzero(val_mask),
        np.flatnonzero(test_mask),
    )


def _iter_training_chunks(processed_data, train_indices, max_chunk_bytes=64 * 1024 * 1024):
    """Yield small advanced-indexing chunks for train-only statistics."""
    sample_nbytes = getattr(processed_data, 'sample_nbytes', None)
    if sample_nbytes is None:
        sample_nbytes = int(processed_data[0].nbytes) if processed_data.size else 0
    chunk_size = max(1, max_chunk_bytes // max(1, sample_nbytes))
    for start in range(0, len(train_indices), chunk_size):
        yield np.abs(processed_data[train_indices[start:start + chunk_size]])


def _fit_xrf55_normalization(processed_data, train_indices, pipeline_steps):
    """Fit XRF55 normalization from bounded-memory training-set chunks."""
    normalize_step = None
    if isinstance(pipeline_steps, dict):
        normalize_step = pipeline_steps.get("normalize")
    if not (
        isinstance(normalize_step, dict)
        and normalize_step.get("method") in {"z-score", "min-max"}
    ):
        return None

    method = normalize_step["method"]
    feature_shape = processed_data.shape[2:]
    count = len(train_indices) * processed_data.shape[1]

    if method == "min-max":
        amin = np.full(feature_shape, np.inf, dtype=np.float64)
        amax = np.full(feature_shape, -np.inf, dtype=np.float64)
        for chunk in _iter_training_chunks(processed_data, train_indices):
            amin = np.minimum(amin, np.min(chunk, axis=(0, 1)))
            amax = np.maximum(amax, np.max(chunk, axis=(0, 1)))
        scale = amax - amin
        scale = np.where(scale < 1e-10, 1.0, scale)
        return {"center": amin, "scale": scale}

    total = np.zeros(feature_shape, dtype=np.float64)
    for chunk in _iter_training_chunks(processed_data, train_indices):
        total += np.sum(chunk, axis=(0, 1), dtype=np.float64)
    mean = total / count

    squared_total = np.zeros(feature_shape, dtype=np.float64)
    for chunk in _iter_training_chunks(processed_data, train_indices):
        squared_total += np.sum((chunk - mean) ** 2, axis=(0, 1), dtype=np.float64)
    std = np.sqrt(squared_total / count)
    std = np.where(std < 1e-10, 1.0, std)
    return {"center": mean, "scale": std}


def _materialize_data_split(
    processed_data,
    labels,
    train_idx,
    val_idx,
    test_idx,
    normalization=None,
):
    """Build array splits, applying optional train-fitted normalization."""
    split_data = [
        processed_data[train_idx],
        processed_data[val_idx],
        processed_data[test_idx],
    ]
    if normalization is not None:
        split_data = [
            (np.abs(data) - normalization["center"]) / normalization["scale"]
            for data in split_data
        ]
    return (
        *split_data,
        labels[train_idx],
        labels[val_idx],
        labels[test_idx],
    )


def _create_xrf55_repetition_split(
    processed_data: np.ndarray,
    labels: np.ndarray,
    repetitions: np.ndarray,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split XRF55 by repetition/trial id for backward compatibility."""
    train_idx, val_idx, test_idx = _create_xrf55_repetition_indices(repetitions)
    normalization = _fit_xrf55_normalization(processed_data, train_idx, pipeline_steps)
    return _materialize_data_split(
        processed_data,
        labels,
        train_idx,
        val_idx,
        test_idx,
        normalization,
    )


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
    model_name: str = "CSIModel",
    model_kwargs: Optional[Dict[str, Any]] = None,
    pipeline_steps: Optional[Dict[str, Dict[str, Any]]] = None,
    algorithm_config_file: Optional[str] = None,
    algorithm_preset: Optional[str] = None,
    reader: Optional[str] = None,
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
    storage_mode: str = "memory",
    storage_dtype: str = "float32",
) -> None:
    """
    Run the full CSI classification pipeline.

    Args:
        input_path: Path to input data directory
        output_folder: Path to output directory
        dataset: Dataset name
        model_path: Optional path to custom model file
        model_name: Registered model name used when model_path is not provided
        model_kwargs: Extra keyword arguments passed to the registered/custom model
        pipeline_steps: Explicit algorithm pipeline steps for ConfigurableProcessor
        algorithm_config_file: YAML/JSON algorithm config file loaded by wsdp.algorithms.load_config
        algorithm_preset: Algorithm preset name loaded by wsdp.algorithms.apply_preset
        reader: Optional reader/dataset name used to select the reader. When None,
            ``dataset`` is used to select the reader.
        batch_size: Override default batch size
        learning_rate: Override default learning rate
        weight_decay: Override default weight decay
        num_epochs: Override default number of epochs
        padding_length: Override default padding length
        test_split: Fraction of data held out (not used for training) (default 0.3)
        val_split: Fraction of held-out data used for validation (default 0.5)
        num_seeds: Number of random seeds to run (default 5)
        config_file: Optional YAML config file to load parameters from
        num_workers: Number of DataLoader workers. When None, uses up to 8.
            Shared-memory and memmap-backed data support spawned workers without
            copying the source arrays.
        progress_callback: Optional callable invoked after each training epoch with a dict of metrics
        use_cache: Cache only when ``storage_mode='disk'`` is explicitly chosen.
            Default shared-memory preprocessing never creates cache files.
        storage_mode: Default ``memory`` stores compact real samples in shared
            memory. ``auto`` is a compatibility alias for ``memory``; ``disk``
            is an explicit opt-in for the legacy disk-shard representation.
        storage_dtype: Shared-result precision, default ``float32``. Algorithms
            and stored model inputs retain their full single-precision values.
            Use ``float16`` to halve shared storage at the cost of lossy
            quantization; batches are always returned as float32 tensors.
    """
    ipath = input_path
    os.makedirs(output_folder, exist_ok=True)
    opath = Path(output_folder)
    dataset_name = dataset

    start_method = mp.get_start_method()
    effective_num_workers = (
        num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
    )

    model_kwargs = model_kwargs or {}
    resolved_pipeline_steps = _resolve_pipeline_steps(
        pipeline_steps=pipeline_steps,
        algorithm_config_file=algorithm_config_file,
        algorithm_preset=algorithm_preset,
    )

    if model_path is not None:
        logger.info(f"Loading model from {model_path}")
    else:
        logger.info(f"Loading registered model: {model_name}")

    if resolved_pipeline_steps is None:
        logger.info("Using default BaseProcessor preprocessing pipeline")
    else:
        logger.info(f"Using configurable preprocessing pipeline: {resolved_pipeline_steps}")

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

    # begin to preprocess, training and eval. Disk caching is strictly opt-in;
    # the default shared-memory path neither reads nor writes processed caches.
    cache_enabled = use_cache and storage_mode == 'disk'
    if use_cache and not cache_enabled:
        logger.info("Shared-memory preprocessing keeps caching disabled")
    cache_dir = os.path.join(output_folder, '.wsdp_cache')
    cached_result = None
    cache_key = None
    if cache_enabled:
        # XRF55 now stores repetition_id as the split group. Add a version marker
        # so old caches with user_id groups are not reused by accident.
        cache_preprocess_config = resolved_pipeline_steps
        if dataset_name == "xrf55":
            cache_preprocess_config = {
                "pipeline_steps": resolved_pipeline_steps,
                "split_protocol": "xrf55_repetition_12_4_4_train_global_normalize",
            }
        elif dataset_name == "widar":
            cache_preprocess_config = {
                "pipeline_steps": resolved_pipeline_steps,
                "split_protocol": "widar_condition_position_orientation_receiver",
                "input_representation": "automatic_amplitude_phase_v1",
            }
        elif uses_phase_amplitude(dataset_name):
            cache_preprocess_config = {
                "pipeline_steps": resolved_pipeline_steps,
                "input_representation": "automatic_amplitude_phase_v1",
            }
        cache_key = get_cache_key(
            ipath,
            dataset_name,
            pad_len,
            preprocess_config=cache_preprocess_config,
        )
        cached_result = load_cache(cache_dir, cache_key)

    if cached_result is not None:
        logger.info("Cache hit: loaded preprocessed data from cache")
        processed_data = cached_result['processed_data']
        zero_indexed_labels = cached_result['labels']
        zero_indexed_groups = cached_result['groups']
        unique_labels = cached_result['unique_labels']
    else:
        if cache_enabled:
            logger.info("Cache miss: processing data from scratch")
            storage_path = os.path.join(cache_dir, f"{cache_key}.npy")
        else:
            storage_path = str(opath / ".wsdp_stream" / "processed.npy")
        processed_data, zero_indexed_labels, zero_indexed_groups, unique_labels = \
            _load_and_preprocess(
                ipath,
                dataset_name,
                pad_len,
                pipeline_steps=resolved_pipeline_steps,
                reader=reader,
                storage_path=storage_path,
                storage_mode=storage_mode,
                storage_dtype=storage_dtype,
            )
        if cache_enabled:
            save_cache(cache_dir, cache_key, processed_data, zero_indexed_labels,
                       zero_indexed_groups, unique_labels)

    disk_backed = isinstance(processed_data, np.memmap) or (
        isinstance(processed_data, ChunkedCSIArray) and processed_data.disk_backed
    )
    shared_backed = (
        isinstance(processed_data, ChunkedCSIArray) and processed_data.shared_backed
    )
    if start_method == "spawn" and not disk_backed and not shared_backed:
        if num_workers is None:
            effective_num_workers = 0
        elif effective_num_workers > 0:
            logger.warning(
                "Spawned DataLoader workers duplicate ordinary in-memory arrays; "
                "use num_workers=0 for externally supplied dense arrays."
            )

    logger.info(f"the following {num_seeds} seeds will be used: {random_seeds}")

    top1_accuracies = []
    seed_records = []

    # Check if we have enough groups for GroupShuffleSplit
    n_groups = len(set(zero_indexed_groups))
    use_simple_split = n_groups < 3
    if use_simple_split:
        logger.warning(f"Only {n_groups} group(s) found (< 3). "
                       f"Using simple train_test_split instead of GroupShuffleSplit.")

    for i, current_seed in enumerate(random_seeds):
        print(f"\n{'=' * 25} epoch {i + 1}/{len(random_seeds)} "
              f"begin (Random State: {current_seed}) {'=' * 25}\n")

        train_idx, val_idx, test_idx, xrf55_protocol = _create_data_split_indices(
            zero_indexed_labels,
            zero_indexed_groups,
            test_split,
            val_split,
            current_seed,
            use_simple_split,
            dataset=dataset_name,
        )
        normalization = None
        if xrf55_protocol:
            normalization = _fit_xrf55_normalization(
                processed_data, train_idx, resolved_pipeline_steps
            )

        logger.info(f"num of samples in train_data: {len(train_idx)}, "
                     f"num of samples in test_data: {len(test_idx)}, num of samples in val_data: {len(val_idx)}")

        train_dataset = CSIDataset(
            processed_data,
            zero_indexed_labels,
            dataset_name=dataset_name,
            pipeline_steps=resolved_pipeline_steps,
            indices=train_idx,
            normalization=normalization,
        )
        test_dataset = CSIDataset(
            processed_data,
            zero_indexed_labels,
            dataset_name=dataset_name,
            pipeline_steps=resolved_pipeline_steps,
            indices=test_idx,
            normalization=normalization,
        )
        val_dataset = CSIDataset(
            processed_data,
            zero_indexed_labels,
            dataset_name=dataset_name,
            pipeline_steps=resolved_pipeline_steps,
            indices=val_idx,
            normalization=normalization,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=effective_num_workers, shuffle=False)

        num_classes = len(unique_labels)
        input_shape = train_dataset.input_shape
        if model_path is None:
            model = create_model(
                model_name,
                num_classes=num_classes,
                input_shape=input_shape,
                **model_kwargs,
            )
        else:
            model = load_custom_model(
                model_path,
                num_classes,
                input_shape=input_shape,
                model_kwargs=model_kwargs,
            )
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

        # ---- collect per-seed metrics ----
        if isinstance(training_history, dict) and training_history.get('train_acc'):
            train_acc = training_history['train_acc'][-1] / 100.0
        else:
            train_acc = 0.0
        val_acc = checkpoint.get('best_val_acc', 0.0) / 100.0
        seed_records.append(SeedRecord(
            seed=current_seed,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=current_top1_acc,
        ))

    accuracies_np = np.array(top1_accuracies)
    mean_accuracy = np.mean(accuracies_np)
    variance_accuracy = np.var(accuracies_np)

    logger.info(f"All {len(random_seeds)} Top-1 acc: {[f'{acc:.4f}' for acc in top1_accuracies]}")
    logger.info(f"Avg Top-1 acc: {mean_accuracy:.4f}")
    logger.info(f"Variance of Top-1 acc: {variance_accuracy:.6f}")

    # ---- persist pipeline record ----
    reader_key = reader or dataset_name
    reader_name = readers.get_reader_class(reader_key).__name__
    if resolved_pipeline_steps is None:
        proc_type = "BaseProcessor"
        proc_steps = {"phase_calibration": "default", "wavelet_denoise_csi": "default"}
    else:
        proc_type = "ConfigurableProcessor"
        proc_steps = resolved_pipeline_steps
    model_str = f"custom:{model_path}" if model_path is not None else model_name

    persist_pipeline_record(
        output_folder=output_folder,
        dataset=dataset_name,
        total_samples=len(processed_data),
        reader_name=reader_name,
        processor_type=proc_type,
        processor_steps=proc_steps,
        model=model_str,
        seed_records=seed_records,
    )

    logger.info("All pipeline complete")
    if hasattr(processed_data, 'close'):
        processed_data.close()
