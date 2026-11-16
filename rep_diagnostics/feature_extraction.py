"""Latent feature extraction from trained checkpoints (torch required).

torch is imported lazily so that metric computation and the test suite
stay CPU/numpy-only. Everything here runs fine on a CPU: it is a single
forward pass over a few thousand samples per checkpoint.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "feature extraction needs PyTorch; install it on the machine "
            "that holds the checkpoints (CPU-only build is sufficient), or "
            "skip this module and feed saved .npz features to run_analysis"
        ) from exc
    return torch


class LatentHook:
    """Capture the output of a chosen submodule during forward passes."""

    def __init__(self, module):
        self._features = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._features = output.detach()

    def get(self):
        return self._features

    def close(self):
        self._handle.remove()


def resolve_latent_layer(model, layer_spec=None):
    """Resolve `layer_spec` (None | name | -k) to a submodule of `model`.

    None picks the penultimate child module (the layer feeding the
    classifier head), matching "latent representation" in the paper.
    A string selects by module name; an int k selects model.children()[-k].
    """
    if layer_spec is None:
        children = list(model.children())
        if len(children) < 2:
            raise ValueError("model has fewer than two child modules; "
                             "specify layer explicitly")
        return children[-2]
    if isinstance(layer_spec, str):
        modules = dict(model.named_modules())
        if layer_spec not in modules:
            raise KeyError(f"no module named {layer_spec!r}; "
                           f"available: {sorted(modules)}")
        return modules[layer_spec]
    if isinstance(layer_spec, int):
        children = list(model.children())
        return children[layer_spec]
    raise TypeError(f"unsupported layer_spec: {layer_spec!r}")


def extract_latent_features(model, data, batch_size=64, device="cpu",
                            layer_spec=None):
    """Forward `data` through `model` and return latent features (N, d).

    `data` is anything indexable that yields model inputs (a Tensor or a
    Dataset whose items are single inputs). Runs under torch.no_grad on
    the requested device; works on CPU without modification.
    """
    torch = _import_torch()
    hook = LatentHook(resolve_latent_layer(model, layer_spec))
    was_training = model.training
    model.eval().to(device)
    feats = []
    try:
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            if not torch.is_tensor(batch):
                batch = torch.stack([data[i] for i in
                                     range(start, min(start + batch_size,
                                                      len(data)))])
            batch = batch.to(device)
            model(batch)
            feats.append(hook.get().cpu().numpy().reshape(len(batch), -1))
    finally:
        hook.close()
        if was_training:
            model.train()
    return np.concatenate(feats, axis=0)


def save_seed_features(checkpoint_path, model, data_by_split, out_dir, seed,
                       batch_size=64, device="cpu", layer_spec=None):
    """Extract and persist per-seed features as features_seed{seed}.npz.

    `data_by_split` maps split names ("train"/"test") to indexable data.
    Each .npz contains X_<split> and, when labels are provided via a
    (`data`, `labels`) tuple, y_<split>.
    """
    os.makedirs(out_dir, exist_ok=True)
    arrays = {}
    for split, payload in data_by_split.items():
        if isinstance(payload, tuple):
            data, labels = payload
        else:
            data, labels = payload, None
        feats = extract_latent_features(model, data, batch_size=batch_size,
                                        device=device, layer_spec=layer_spec)
        arrays[f"X_{split}"] = feats
        if labels is not None:
            arrays[f"y_{split}"] = np.asarray(labels)
    out_path = os.path.join(out_dir, f"features_seed{seed}.npz")
    np.savez(out_path, **arrays)
    return out_path
