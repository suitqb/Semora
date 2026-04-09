"""Analysis utilities shared across extraction and complexity modes."""

from __future__ import annotations

import math
from pathlib import Path


def assemble_report(
    ordered_paths: list[Path],
    output_dir: Path,
    title: str = "Semora Report",
) -> None:
    """Stitch a list of PNG paths into a single report.png.

    Args:
        ordered_paths: PNGs in the order they should appear (left-to-right, top-to-bottom).
        output_dir:    Directory where report.png is saved.
        title:         Figure suptitle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401 — needed by plt.imread

    paths = [p for p in ordered_paths if p is not None and p.exists()]
    if not paths:
        return

    imgs = [(p.stem, plt.imread(str(p))) for p in paths]
    n = len(imgs)
    fig_w = 20.0

    # Use each image's actual aspect ratio as height_ratio → no padding between panels
    height_ratios = [img.shape[0] / img.shape[1] for _, img in imgs]
    fig_h = fig_w * sum(height_ratios)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.03},
    )
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e0e3f0", y=1.0)

    for i, (stem, img) in enumerate(imgs):
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

    plt.subplots_adjust(top=0.99, bottom=0, left=0, right=1)
    out = output_dir / "report.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")
