"""
Utility functions for Multi-Model Trading Architecture.

This module provides common utilities for:
    - Random seed management
    - Device detection
    - Logging utilities
    - Progress tracking

Author: Trading Team
Date: 2025-12-29
"""

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """
    Get compute device.

    Args:
        device: Preferred device ('cuda', 'cpu', or None for auto).

    Returns:
        Device string ('cuda' or 'cpu').
    """
    if device is not None:
        return device

    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_info() -> dict:
    """
    Get GPU information.

    Returns:
        Dictionary with GPU info (name, memory, etc.).
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
    }


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def format_number(n: float, precision: int = 2) -> str:
    """
    Format number with K/M/B suffixes.

    Args:
        n: Number to format.
        precision: Decimal precision.

    Returns:
        Formatted string.
    """
    if abs(n) >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


class ProgressTracker:
    """
    Simple progress tracker for training loops.

    Tracks metrics and prints periodic updates.
    """

    def __init__(self, total: int, name: str = "Progress", print_every: int = 10):
        """
        Initialize tracker.

        Args:
            total: Total number of steps.
            name: Name for display.
            print_every: Print frequency (percentage).
        """
        self.total = total
        self.name = name
        self.print_every = print_every
        self.current = 0
        self.last_printed = -1
        self.metrics = {}

    def update(self, n: int = 1, **metrics) -> None:
        """
        Update progress.

        Args:
            n: Number of steps completed.
            **metrics: Metric values to track.
        """
        self.current += n

        # Update running metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # Print if needed
        pct = int(100 * self.current / self.total)
        if pct >= self.last_printed + self.print_every:
            self.last_printed = pct
            self._print_status()

    def _print_status(self) -> None:
        """Print current status."""
        pct = 100 * self.current / self.total
        status = f"{self.name}: {self.current}/{self.total} ({pct:.0f}%)"

        # Add average metrics
        metric_strs = []
        for key, values in self.metrics.items():
            avg = np.mean(values[-100:])  # Average of last 100
            metric_strs.append(f"{key}={avg:.4f}")

        if metric_strs:
            status += " | " + " | ".join(metric_strs)

        print(status)

    def finish(self) -> dict:
        """
        Finish tracking and return final metrics.

        Returns:
            Dictionary with final metric averages.
        """
        final = {}
        for key, values in self.metrics.items():
            final[key] = np.mean(values)
            final[f"{key}_final"] = values[-1] if values else 0

        print(f"{self.name}: Complete!")
        return final
