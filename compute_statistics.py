#!/usr/bin/env python3
"""
Statistical significance testing for Faroese POS tagger model comparisons.

Performs:
  - Paired Wilcoxon signed-rank tests between key model pairs
  - Nadeau-Bengio corrected paired t-tests (accounts for CV dependency)
  - Holm-Bonferroni correction for multiple comparisons
  - Cohen's d effect size

Outputs a JSON file with all results and prints a human-readable summary table.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_FOLDS = 10

# For 10-fold CV: each fold uses 1/10 as test, 9/10 as train.
N_TEST_OVER_N_TRAIN = 1.0 / 9.0

SCANDIBERT_MODELS = [
    "singlelabel",
    "multilabel",
    "multilabel_unconstrained_normalized",
    "multilabel_unconstrained_unnormalized",
]

# Human-readable short names used in tables and JSON output.
MODEL_DISPLAY_NAMES = {
    "singlelabel": "Single",
    "multilabel": "Multi (constr.)",
    "multilabel_unconstrained_normalized": "Multi (unconstr. norm)",
    "multilabel_unconstrained_unnormalized": "Multi (unconstr. unnorm)",
    "tnt": "TnT",
    "abltagger": "ABLTagger",
}

# The seven pairwise comparisons we care about.
COMPARISONS: List[Tuple[str, str]] = [
    ("multilabel", "singlelabel"),
    ("multilabel", "multilabel_unconstrained_normalized"),
    ("multilabel", "multilabel_unconstrained_unnormalized"),
    ("multilabel", "abltagger"),
    ("multilabel", "tnt"),
    ("singlelabel", "abltagger"),
    ("multilabel_unconstrained_normalized", "multilabel_unconstrained_unnormalized"),
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict:
    """Load a JSON file and return its contents as a dict."""
    with open(path, "r") as f:
        return json.load(f)


def load_scandibert_val(results_dir: Path, model: str) -> List[float]:
    """Load per-fold validation composite accuracy for a ScandiBERT model."""
    accs = []
    for i in range(NUM_FOLDS):
        path = results_dir / f"fold_{i}_{model}" / "best_results.json"
        data = load_json(path)
        accs.append(float(data["val_composite_accuracy"]))
    return accs


def load_scandibert_ood(results_dir: Path, model: str) -> List[float]:
    """Load per-fold OOD composite accuracy for a ScandiBERT model."""
    accs = []
    for i in range(NUM_FOLDS):
        path = results_dir / f"fold_{i}_{model}" / "ood_results.json"
        data = load_json(path)
        accs.append(float(data["composite_accuracy"]))
    return accs


def load_tnt_val(results_dir: Path) -> List[float]:
    """Load per-fold TnT validation composite accuracy."""
    accs = []
    for i in range(NUM_FOLDS):
        path = results_dir / f"fold_{i}_tnt" / "results.json"
        data = load_json(path)
        accs.append(float(data["val_composite_accuracy"]))
    return accs


def load_tnt_ood(results_dir: Path) -> List[float]:
    """Load per-fold TnT OOD composite accuracy."""
    accs = []
    for i in range(NUM_FOLDS):
        path = results_dir / f"fold_{i}_tnt" / "ood_results.json"
        data = load_json(path)
        accs.append(float(data["composite_accuracy"]))
    return accs


def load_abltagger_val(summary_path: Path) -> List[float]:
    """
    Load per-fold ABLTagger validation accuracy from the summary JSON.

    NOTE: ABLTagger accuracy is on a 0-100 scale; we convert to 0-1.
    """
    data = load_json(summary_path)
    accs = []
    for fold_entry in data["folds"]:
        acc_pct = float(fold_entry["metrics"]["fine_grained"]["accuracy"])
        accs.append(acc_pct / 100.0)
    return accs


def load_abltagger_ood(summary_path: Path) -> List[float]:
    """
    Load per-model ABLTagger OOD accuracy from the OOD summary JSON.

    NOTE: ABLTagger accuracy is on a 0-100 scale; we convert to 0-1.
    """
    data = load_json(summary_path)
    accs = []
    for model_entry in data["models"]:
        acc_pct = float(model_entry["metrics"]["fine_grained"]["accuracy"])
        accs.append(acc_pct / 100.0)
    return accs


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test (two-sided).

    Returns (statistic, p_value).  If all differences are zero the test is
    undefined -- we return (nan, 1.0).
    """
    diff = a - b
    if np.all(diff == 0):
        return (float("nan"), 1.0)
    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        # scipy raises ValueError when all differences are zero after
        # rounding or when n is too small.
        return (float("nan"), 1.0)
    return (float(stat), float(p))


def nadeau_bengio_corrected_t_test(
    a: np.ndarray,
    b: np.ndarray,
    k: int = NUM_FOLDS,
    n_test_over_n_train: float = N_TEST_OVER_N_TRAIN,
) -> Tuple[float, float]:
    """
    Nadeau & Bengio (2003) corrected resampled paired t-test.

    The standard paired-samples variance is multiplied by
    (1/k + n_test/n_train) to account for the non-independence of CV folds.
    We then compute a t-statistic and use k-1 degrees of freedom.

    Returns (t_statistic, p_value).
    """
    diff = a - b
    mean_diff = np.mean(diff)
    # Variance of the differences (Bessel-corrected).
    var_diff = np.var(diff, ddof=1)

    if var_diff == 0:
        return (float("nan"), 1.0)

    # Corrected variance per Nadeau & Bengio.
    corrected_var = var_diff * (1.0 / k + n_test_over_n_train)
    corrected_se = math.sqrt(corrected_var)

    t_stat = mean_diff / corrected_se
    df = k - 1
    p_value = 2.0 * stats.t.sf(abs(t_stat), df)  # two-sided

    return (float(t_stat), float(p_value))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples.

    d = mean(a - b) / pooled_std, where pooled_std is the pooled standard
    deviation of the two samples (not of the differences).
    """
    n_a = len(a)
    n_b = len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return float("nan")
    return float(np.mean(a - b) / pooled_std)


def classify_effect_size(d: float) -> str:
    """Classify Cohen's d magnitude using standard thresholds."""
    d_abs = abs(d)
    if math.isnan(d_abs):
        return "undefined"
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_all_model_data(
    results_dir: Path,
    abltagger_val_path: Optional[Path],
    abltagger_ood_path: Optional[Path],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Load per-fold accuracies for every model.

    Returns:
        {model_name: {"val": [...], "ood": [...]}}
    """
    data: Dict[str, Dict[str, List[float]]] = {}

    # ScandiBERT models
    for model in SCANDIBERT_MODELS:
        data[model] = {
            "val": load_scandibert_val(results_dir, model),
            "ood": load_scandibert_ood(results_dir, model),
        }

    # TnT
    data["tnt"] = {
        "val": load_tnt_val(results_dir),
        "ood": load_tnt_ood(results_dir),
    }

    # ABLTagger
    if abltagger_val_path and abltagger_ood_path:
        data["abltagger"] = {
            "val": load_abltagger_val(abltagger_val_path),
            "ood": load_abltagger_ood(abltagger_ood_path),
        }
    else:
        print(
            "WARNING: ABLTagger summary paths not provided; "
            "comparisons involving ABLTagger will be skipped.",
            file=sys.stderr,
        )

    return data


def run_comparisons(
    model_data: Dict[str, Dict[str, List[float]]],
) -> Dict:
    """
    Run all pairwise comparisons and collect raw + corrected p-values.

    Returns a dict suitable for JSON serialisation.
    """
    # We will collect results for both evaluation settings (val and OOD).
    all_results: Dict[str, list] = {"val": [], "ood": []}

    for eval_setting in ("val", "ood"):
        for model_a, model_b in COMPARISONS:
            # Skip comparisons where one of the models was not loaded.
            if model_a not in model_data or model_b not in model_data:
                continue

            a = np.array(model_data[model_a][eval_setting])
            b = np.array(model_data[model_b][eval_setting])

            mean_a = float(np.mean(a))
            mean_b = float(np.mean(b))
            mean_diff = float(np.mean(a - b))

            w_stat, w_p = wilcoxon_test(a, b)
            nb_t, nb_p = nadeau_bengio_corrected_t_test(a, b)
            d = cohens_d(a, b)

            all_results[eval_setting].append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "display_a": MODEL_DISPLAY_NAMES[model_a],
                    "display_b": MODEL_DISPLAY_NAMES[model_b],
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_diff,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p": w_p,
                    "nb_t_statistic": nb_t,
                    "nb_p": nb_p,
                    "cohens_d": d,
                    "effect_magnitude": classify_effect_size(d),
                }
            )

    # ---- Holm-Bonferroni correction across all comparisons per setting ----
    for eval_setting in ("val", "ood"):
        entries = all_results[eval_setting]
        if not entries:
            continue

        # Correct Wilcoxon p-values.
        w_pvals = np.array([e["wilcoxon_p"] for e in entries])
        _, w_corrected, _, _ = multipletests(w_pvals, method="holm")
        for entry, cp in zip(entries, w_corrected):
            entry["wilcoxon_p_corrected"] = float(cp)

        # Correct Nadeau-Bengio p-values.
        nb_pvals = np.array([e["nb_p"] for e in entries])
        _, nb_corrected, _, _ = multipletests(nb_pvals, method="holm")
        for entry, cp in zip(entries, nb_corrected):
            entry["nb_p_corrected"] = float(cp)

    return all_results


def print_summary_table(results: Dict) -> None:
    """Print a human-readable summary table to stdout."""
    for eval_setting in ("val", "ood"):
        entries = results.get(eval_setting, [])
        if not entries:
            continue

        setting_label = "Validation (In-Domain)" if eval_setting == "val" else "Out-of-Domain (OOD)"
        print(f"\n{'=' * 120}")
        print(f"  {setting_label}")
        print(f"{'=' * 120}")

        header = (
            f"{'Comparison':<50s}  "
            f"{'Mean A':>8s}  {'Mean B':>8s}  {'Diff':>8s}  "
            f"{'Wilcox p':>9s}  {'W p(corr)':>9s}  "
            f"{'NB p':>9s}  {'NB p(corr)':>10s}  "
            f"{'Cohen d':>8s}  {'Effect':>12s}"
        )
        print(header)
        print("-" * 120)

        for e in entries:
            name = f"{e['display_a']} vs {e['display_b']}"
            sig_w = "*" if e["wilcoxon_p_corrected"] < 0.05 else ""
            sig_nb = "*" if e["nb_p_corrected"] < 0.05 else ""
            print(
                f"{name:<50s}  "
                f"{e['mean_a'] * 100:>7.2f}%  {e['mean_b'] * 100:>7.2f}%  "
                f"{e['mean_diff'] * 100:>+7.3f}%  "
                f"{e['wilcoxon_p']:>9.5f}  {e['wilcoxon_p_corrected']:>8.5f}{sig_w:<1s}  "
                f"{e['nb_p']:>9.5f}  {e['nb_p_corrected']:>9.5f}{sig_nb:<1s}  "
                f"{e['cohens_d']:>+8.3f}  {e['effect_magnitude']:>12s}"
            )

        print()

    # Legend
    print("* = significant at alpha=0.05 after Holm-Bonferroni correction")
    print(f"Nadeau-Bengio corrected t-test uses k={NUM_FOLDS} folds, "
          f"n_test/n_train={N_TEST_OVER_N_TRAIN:.4f}")
    print()


def build_summary_block(results: Dict) -> Dict:
    """Build a top-level summary with model means and stds."""
    summary: Dict[str, Dict] = {}
    for eval_setting in ("val", "ood"):
        entries = results.get(eval_setting, [])
        # Collect unique models mentioned in results.
        models_seen: Dict[str, float] = {}
        for e in entries:
            models_seen[e["model_a"]] = e["mean_a"]
            models_seen[e["model_b"]] = e["mean_b"]
        summary[eval_setting] = {
            MODEL_DISPLAY_NAMES.get(m, m): round(v * 100, 2)
            for m, v in models_seen.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical significance testing for Faroese POS tagger models."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/all_splits_eval_bragd_adamw",
        help="Directory containing per-fold model results (default: all_splits_eval_bragd_adamw)",
    )
    parser.add_argument(
        "--abltagger-val-summary",
        type=str,
        default="results/abltagger/val_summary.json",
        help="Path to ABLTagger validation summary.json",
    )
    parser.add_argument(
        "--abltagger-ood-summary",
        type=str,
        default="results/abltagger/ood_summary.json",
        help="Path to ABLTagger OOD ood_summary.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="statistics_results.json",
        help="Output JSON file path (default: statistics_results.json)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    abltagger_val_path = Path(args.abltagger_val_summary) if args.abltagger_val_summary else None
    abltagger_ood_path = Path(args.abltagger_ood_summary) if args.abltagger_ood_summary else None

    if abltagger_val_path and not abltagger_val_path.is_file():
        print(f"ERROR: ABLTagger val summary not found: {abltagger_val_path}", file=sys.stderr)
        sys.exit(1)
    if abltagger_ood_path and not abltagger_ood_path.is_file():
        print(f"ERROR: ABLTagger OOD summary not found: {abltagger_ood_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Load data ----
    print("Loading per-fold results ...")
    model_data = load_all_model_data(results_dir, abltagger_val_path, abltagger_ood_path)

    # Print quick sanity check: per-model means.
    print("\nPer-model mean composite accuracy (%):")
    for model_name, scores in model_data.items():
        val_mean = np.mean(scores["val"]) * 100
        ood_mean = np.mean(scores["ood"]) * 100
        val_std = np.std(scores["val"], ddof=1) * 100
        ood_std = np.std(scores["ood"], ddof=1) * 100
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"  {display:<30s}  val={val_mean:6.2f} +/- {val_std:5.2f}    "
              f"ood={ood_mean:6.2f} +/- {ood_std:5.2f}")

    # ---- Run comparisons ----
    print("\nRunning statistical tests ...")
    results = run_comparisons(model_data)

    # ---- Print table ----
    print_summary_table(results)

    # ---- Build output JSON ----
    output = {
        "description": (
            "Statistical significance tests for Faroese POS tagger model comparisons. "
            "Includes Wilcoxon signed-rank tests, Nadeau-Bengio corrected paired t-tests, "
            "Holm-Bonferroni corrected p-values, and Cohen's d effect sizes."
        ),
        "parameters": {
            "num_folds": NUM_FOLDS,
            "n_test_over_n_train": N_TEST_OVER_N_TRAIN,
            "correction_method": "holm-bonferroni",
            "results_dir": str(results_dir),
            "abltagger_val_summary": str(abltagger_val_path) if abltagger_val_path else None,
            "abltagger_ood_summary": str(abltagger_ood_path) if abltagger_ood_path else None,
        },
        "model_means_pct": build_summary_block(results),
        "comparisons": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
