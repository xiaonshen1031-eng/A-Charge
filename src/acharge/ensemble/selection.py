from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ttest_rel


COMPLEXITY_ORDER = {"SB": 0, "HB": 1, "RSF": 2, "LAR": 3, "NAR": 4}


def holm_rejections(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rejected = {name: False for name in p_values}
    for rank, (name, p_value) in enumerate(ordered):
        threshold = alpha / (len(ordered) - rank)
        if p_value <= threshold:
            rejected[name] = True
        else:
            break
    return rejected


def select_fusion_candidate(
    results: dict[str, dict[str, Any]],
    *,
    noninferiority_margin: float = 5e-4,
    holm_alpha: float = 0.05,
    block_length: int = 24,
) -> dict[str, Any]:
    if not results:
        raise ValueError("fusion result set is empty")
    best = min(results, key=lambda name: (float(results[name]["macro_R_cal"]), COMPLEXITY_ORDER.get(name, 99), name))
    best_loss = np.asarray(results[best]["per_origin_loss"], dtype=float)
    p_values: dict[str, float] = {}
    block_means: dict[str, list[float]] = {}
    for name, result in results.items():
        loss = np.asarray(result["per_origin_loss"], dtype=float)
        n = min(len(loss), len(best_loss))
        n_blocks = n // block_length
        if n_blocks < 2:
            raise ValueError("insufficient CAL OOF origins for block analysis")
        candidate_blocks = loss[: n_blocks * block_length].reshape(n_blocks, block_length).mean(axis=1)
        reference_blocks = best_loss[: n_blocks * block_length].reshape(n_blocks, block_length).mean(axis=1)
        block_means[name] = candidate_blocks.tolist()
        if name == best:
            p_values[name] = 1.0
        else:
            # Reject non-inferiority only when candidate loss exceeds best+margin.
            difference = candidate_blocks - noninferiority_margin - reference_blocks
            if float(np.std(difference)) <= 1e-15:
                p_values[name] = 0.0 if float(np.mean(difference)) > 0.0 else 1.0
            else:
                p_values[name] = float(ttest_rel(
                    candidate_blocks - noninferiority_margin,
                    reference_blocks,
                    alternative="greater",
                ).pvalue)
    tested = {name: value for name, value in p_values.items() if name != best}
    rejected = holm_rejections(tested, alpha=holm_alpha)
    rejected[best] = False
    mean_best = float(results[best]["macro_R_cal"])
    eligible = [
        name for name, result in results.items()
        if float(result["macro_R_cal"]) <= mean_best + noninferiority_margin and not rejected.get(name, False)
    ]
    if not eligible:
        eligible = [best]
    selected = min(eligible, key=lambda name: (COMPLEXITY_ORDER.get(name, 99), float(results[name]["macro_R_cal"]), name))
    return {
        "selected_fusion": selected,
        "best_mean_fusion": best,
        "eligible_noninferior": eligible,
        "noninferiority_margin": float(noninferiority_margin),
        "holm_alpha": float(holm_alpha),
        "block_length": int(block_length),
        "one_sided_p_values": p_values,
        "holm_reject_noninferiority": rejected,
        "complexity_order": COMPLEXITY_ORDER,
        "block_mean_losses": block_means,
    }
