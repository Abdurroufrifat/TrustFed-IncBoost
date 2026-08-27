from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    array = np.asarray(matrix)
    figure_width = max(6, min(14, 0.8 * len(class_names) + 3))
    fig, ax = plt.subplots(figsize=(figure_width, figure_width * 0.8))
    sns.heatmap(
        array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
