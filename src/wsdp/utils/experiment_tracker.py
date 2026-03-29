"""Lightweight experiment tracking for WSDP.

Supports three backends:
- 'local': Writes metrics to CSV (always available, no dependencies)
- 'wandb': Weights & Biases (requires ``pip install wandb``)
- 'mlflow': MLflow (requires ``pip install mlflow``)
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface.

    Parameters
    ----------
    backend : str
        One of ``'local'``, ``'wandb'``, or ``'mlflow'``.
    project_name : str
        Logical project name (used by W&B / MLflow).
    run_name : str or None
        Human-readable run name.  Auto-generated when *None*.
    output_dir : str or None
        Directory for local CSV output (local backend only).
        Defaults to ``'./experiments'``.
    **kwargs
        Extra keyword arguments forwarded to the backend's ``init`` call.
    """

    def __init__(
        self,
        backend: str = "local",
        project_name: str = "wsdp",
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.backend = backend
        self.project_name = project_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir or "./experiments")
        self._params: Dict[str, Any] = {}
        self._metrics_rows: list = []
        self._run = None  # backend-specific run object

        if backend == "local":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.output_dir / f"{self.run_name}_metrics.csv"
            logger.info("ExperimentTracker: using local CSV backend -> %s", self._csv_path)

        elif backend == "wandb":
            try:
                import wandb  # noqa: F811

                self._run = wandb.init(
                    project=project_name,
                    name=self.run_name,
                    **kwargs,
                )
                logger.info("ExperimentTracker: using W&B backend")
            except ImportError:
                logger.warning(
                    "wandb is not installed. Falling back to local CSV backend. "
                    "Install with: pip install wandb"
                )
                self.backend = "local"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._csv_path = self.output_dir / f"{self.run_name}_metrics.csv"

        elif backend == "mlflow":
            try:
                import mlflow  # noqa: F811

                mlflow.set_experiment(project_name)
                self._run = mlflow.start_run(run_name=self.run_name, **kwargs)
                logger.info("ExperimentTracker: using MLflow backend")
            except ImportError:
                logger.warning(
                    "mlflow is not installed. Falling back to local CSV backend. "
                    "Install with: pip install mlflow"
                )
                self.backend = "local"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._csv_path = self.output_dir / f"{self.run_name}_metrics.csv"
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: local, wandb, mlflow")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters / config values."""
        self._params.update(params)

        if self.backend == "wandb":
            import wandb

            wandb.config.update(params, allow_val_change=True)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_params(params)
        else:
            logger.info("Params logged: %s", params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of numeric metrics for an optional *step*."""
        row = {"step": step, **metrics}
        self._metrics_rows.append(row)

        if self.backend == "wandb":
            import wandb

            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        else:
            # Append to CSV immediately so data is not lost on crash.
            self._write_csv_row(row)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log a file artifact (model checkpoint, plot, etc.)."""
        if self.backend == "wandb":
            import wandb

            artifact = wandb.Artifact(name or Path(path).stem, type="output")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_artifact(path)
        else:
            logger.info("Artifact recorded (local): %s", path)

    def finish(self) -> None:
        """Finalise the current run and flush all data."""
        if self.backend == "wandb" and self._run is not None:
            self._run.finish()
        elif self.backend == "mlflow" and self._run is not None:
            import mlflow

            mlflow.end_run()
        else:
            # Write params summary for local backend.
            if self._params:
                params_path = self.output_dir / f"{self.run_name}_params.csv"
                with open(params_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["param", "value"])
                    for k, v in self._params.items():
                        writer.writerow([k, v])
                logger.info("Params saved to %s", params_path)

        logger.info("ExperimentTracker: run '%s' finished.", self.run_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, row: dict) -> None:
        """Append a single row to the local CSV file."""
        file_exists = self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
