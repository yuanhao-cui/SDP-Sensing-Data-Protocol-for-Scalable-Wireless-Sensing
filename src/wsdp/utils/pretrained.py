"""Pretrained weight management for WSDP models."""

import os
import logging

logger = logging.getLogger(__name__)

PRETRAINED_REGISTRY = {
    # Will be populated as official benchmarks are run
    # Format: 'model_dataset': {'url': '...', 'md5': '...', 'accuracy': 0.0}
}


def list_pretrained():
    """List available pretrained weights.

    Returns:
        dict: Copy of the pretrained registry mapping model_dataset keys
              to their metadata (url, md5, accuracy).
    """
    return dict(PRETRAINED_REGISTRY)


def download_pretrained(model_name, dataset, cache_dir=None):
    """Download pretrained weights for a model+dataset combination.

    Args:
        model_name: Name of the model architecture.
        dataset: Name of the dataset the weights were trained on.
        cache_dir: Optional directory to store downloaded weights.
                   Defaults to ``~/.wsdp/pretrained/`` if not specified.

    Returns:
        str: Path to the downloaded weights file.

    Raises:
        ValueError: If no pretrained weights exist for the given combination.
        NotImplementedError: Download logic is not yet implemented.
    """
    key = f"{model_name}_{dataset}"
    if key not in PRETRAINED_REGISTRY:
        available = list(PRETRAINED_REGISTRY.keys()) or ['none yet']
        raise ValueError(f"No pretrained weights for '{key}'. Available: {available}")

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".wsdp", "pretrained")
    os.makedirs(cache_dir, exist_ok=True)

    # Placeholder for future download logic
    raise NotImplementedError(
        "Pretrained weights will be available after official benchmarks are run."
    )
