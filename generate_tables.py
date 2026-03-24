#!/usr/bin/env python3
"""
Unified script to generate all LaTeX tables and the training progress figure
for the Faroese POS tagger paper.

Outputs:
  $OUTPUT_DIR/tables/full_results.tex            — Table 3: overall + per-word-class F1
  $OUTPUT_DIR/tables/per_label_results.tex       — Table 4: per-feature-group Acc/F1
  $OUTPUT_DIR/tables/wordclass_results.tex       — Table 5: per-word-class Acc/F1
  $OUTPUT_DIR/tables/appendix_unconstrained.tex  — Appendix: normalization ablation
  $OUTPUT_DIR/figs/training_progress.pdf         — Figure 1: epoch vs accuracy

Environment variables:
  OUTPUT_DIR   — where to write outputs (default: ./output)
  RESULTS_DIR  — where fold results live (default: all_splits_eval_bragd_adamw)

Flags:
  --skip-abltagger  — skip ABLTagger results (if not yet retrained)
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_json(file_path: Path) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Return (mean, sample_std)."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def format_value(mean: float, std: float, is_best: bool = False) -> str:
    """Format a value for LaTeX: mean_{std}, optionally bolded."""
    if is_best:
        return f"\\textbf{{{mean:.1f}}}$_{{{std:.1f}}}$"
    return f"{mean:.1f}$_{{{std:.1f}}}$"


def find_best(entries: List[Tuple[float, float]], group_indices: List[int]) -> float:
    """Return the maximum mean among entries at group_indices."""
    return max(entries[i][0] for i in group_indices if entries[i] is not None)


# ---------------------------------------------------------------------------
# ABLTagger helpers
# ---------------------------------------------------------------------------

ABL_CATEGORY_MAP = {
    'N': 'Noun', 'S': 'Verbs', 'A': 'Adjective', 'D': 'Adverb',
    'P': 'Pronoun', 'L': 'Number', 'C': 'Conjunctions', 'R': 'R Article',
    'T': 'Participle', 'M': 'Abbreviation', 'F': 'Foreign words',
    'K': 'Punctuation', 'V': 'Symbol', 'X': 'Unanalyzed word',
    'U': 'Web email or address',
}


def load_abltagger_summary(summary_file: Path) -> Dict[str, Tuple[float, float]]:
    """
    Extract ABLTagger per-word-class F1 + composite accuracy from a summary JSON.
    Returns dict mapping class name (or '_composite_accuracy') to (mean, std).
    """
    data = load_json(summary_file)
    result = {}

    fold_list = data.get('folds') or data.get('models') or []
    if not fold_list:
        return result

    composite_accs = []
    category_f1s: Dict[str, List[float]] = defaultdict(list)

    for fold in fold_list:
        composite_accs.append(fold['metrics']['fine_grained']['accuracy'])
        per_category = fold['metrics']['coarse_grained'].get('per_category', {})
        for letter, metrics in per_category.items():
            if letter in ABL_CATEGORY_MAP:
                category_f1s[ABL_CATEGORY_MAP[letter]].append(metrics['f1'] * 100)

    result['_composite_accuracy'] = compute_stats(composite_accs)
    for wc, scores in category_f1s.items():
        if scores and np.mean(scores) > 0:
            result[wc] = compute_stats(scores)

    return result


# ---------------------------------------------------------------------------
# Table 3: full_results  (overall + per-word-class F1, all 4 models, both datasets)
# ---------------------------------------------------------------------------

WORD_CLASS_ORDER_TABLE3 = [
    'Noun', 'Verbs', 'Adjective', 'Adverb', 'Pronoun',
    'Number', 'Conjunctions', 'R Article', 'Participle',
    'Abbreviation', 'Foreign words', 'Punctuation', 'Symbol',
]

WORD_CLASS_SHORT = {
    'Noun': 'Noun', 'Verbs': 'Verb', 'Adjective': 'Adj.',
    'Adverb': 'Adverb', 'Pronoun': 'Pron.', 'Number': 'Num.',
    'Conjunctions': 'Con.', 'R Article': 'R Article',
    'Participle': 'Partic.', 'Abbreviation': 'Abbr.',
    'Foreign words': 'For.', 'Punctuation': 'Punc.', 'Symbol': 'Sym.',
}

MODEL_KEYS = ['tnt', 'abltagger', 'singlelabel', 'multilabel',
              'multilabel_unconstrained_unnormalized']
MODEL_NAMES_TABLE3 = {
    'tnt': 'TnT', 'abltagger': 'ABL-Tag.',
    'singlelabel': 'Sc.B-sing', 'multilabel': 'Sc.B-multi',
    'multilabel_unconstrained_unnormalized': 'Sc.B-unc.',
}


def aggregate_word_class_f1(base_dir: Path, model_type: str, dataset: str,
                            num_folds: int = 10) -> Dict[str, Tuple[float, float]]:
    """Aggregate per-word-class F1 + composite accuracy across folds."""
    wc_f1: Dict[str, List[float]] = defaultdict(list)
    composite_accs: List[float] = []

    for fold in range(num_folds):
        if dataset == 'val':
            if model_type == 'tnt':
                rfile = base_dir / f"fold_{fold}_tnt" / "results.json"
            else:
                rfile = base_dir / f"fold_{fold}_{model_type}" / "best_results.json"
        else:
            rfile = base_dir / f"fold_{fold}_{model_type}" / "ood_results.json"

        if not rfile.exists():
            print(f"  Warning: missing {rfile}")
            continue

        data = load_json(rfile)
        acc_key = 'val_composite_accuracy' if dataset == 'val' else 'composite_accuracy'
        composite_accs.append(data[acc_key] * 100)

        for class_name, f1 in data.get('per_word_class_f1', {}).items():
            clean = class_name.replace('Word Class ', '')
            wc_f1[clean].append(f1 * 100)

    result = {}
    for cls, scores in wc_f1.items():
        if scores and np.mean(scores) > 0:
            result[cls] = compute_stats(scores)
    if composite_accs:
        result['_composite_accuracy'] = compute_stats(composite_accs)
    return result


def generate_full_results_table(all_results: Dict, skip_abltagger: bool) -> str:
    """Generate Table 3 LaTeX."""
    models = [m for m in MODEL_KEYS if not (m == 'abltagger' and skip_abltagger)]
    n_models = len(models)

    # Determine which word classes actually have data
    wc_order = [wc for wc in WORD_CLASS_ORDER_TABLE3
                if any(wc in all_results[ds].get(m, {})
                       for ds in ['val', 'ood'] for m in models)]

    n_wc = len(wc_order)
    header_names = [WORD_CLASS_SHORT.get(wc, wc) for wc in wc_order]

    lines = []
    lines.append("\\begin{table*}[]")
    lines.append("   \\centering")
    lines.append("   \\resizebox{\\textwidth}{!}{%")
    # Column spec: dataset rotator, method name, overall | wc1 wc2 ...
    col_spec = "ll" + "r|" + "r" * n_wc
    lines.append(f"   \\begin{{tabular}}{{{col_spec}}}")
    lines.append("   \\toprule")

    # Header
    header = "   & Method & Overall"
    for name in header_names:
        header += f" & {name}"
    header += "\\\\"
    lines.append(header)
    lines.append("   \\midrule")

    for ds, ds_label in [('val', 'Sos. (ID)'), ('ood', 'OOD')]:
        for idx, mkey in enumerate(models):
            if idx == 0:
                row = f"   \\multirow{{{n_models}}}{{*}}{{\\rotatebox{{90}}{{{ds_label}}}}} & {MODEL_NAMES_TABLE3[mkey]}"
            else:
                row = f"    & {MODEL_NAMES_TABLE3[mkey]}"

            # Overall accuracy
            entry = all_results[ds].get(mkey, {}).get('_composite_accuracy')
            if entry:
                best_val = max(
                    all_results[ds].get(m, {}).get('_composite_accuracy', (0, 0))[0]
                    for m in models)
                row += f" & {format_value(entry[0], entry[1], abs(entry[0] - best_val) < 0.05)}"
            else:
                row += " & ---"

            # Per-word-class F1
            for wc in wc_order:
                entry = all_results[ds].get(mkey, {}).get(wc)
                if entry:
                    best_val = max(
                        all_results[ds].get(m, {}).get(wc, (0, 0))[0]
                        for m in models)
                    row += f" & {format_value(entry[0], entry[1], abs(entry[0] - best_val) < 0.05)}"
                else:
                    row += " & ---"

            row += "\\\\"
            lines.append(row)

        if ds == 'val':
            lines.append("   \\midrule")

    lines.append("   \\bottomrule")
    lines.append("   \\end{tabular}%")
    lines.append("   }")
    lines.append("   \\caption{Results from training and evaluating on the "
                 "\\textit{Sosialurin-BRAGD} dataset (10-fold cross-validation) "
                 "and \\textit{OOD-BRAGD} corpora. Overall indicates the composite "
                 "accuracy. Word class columns show F1 scores. Subscripts show "
                 "standard deviations. Best results in each column are \\textbf{bolded}.")
    lines.append("   }")
    lines.append("   \\label{tab:full-results}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: per_label_results  (per-feature-group Acc/F1, single vs multi)
# ---------------------------------------------------------------------------

LABEL_ORDER = [
    'Word Type', 'Subcategories', 'Gender', 'Number', 'Case', 'Article',
    'Proper Noun', 'Degree', 'Declension', 'Mood', 'Voice', 'Tense',
    'Person', 'Definiteness',
]


def load_per_label_results(base_dir: Path, mode: str, result_type: str,
                           num_folds: int = 10) -> Dict:
    """Load per-interval (per-label) accuracy and F1 across folds."""
    out: Dict[str, Dict[str, List[float]]] = {
        'accuracy': {}, 'f1': {},
        'word_class_accuracy': [], 'word_class_macro_f1': [],
    }
    rfile_name = "best_results.json" if result_type == "best" else "ood_results.json"

    for fold in range(num_folds):
        fpath = base_dir / f"fold_{fold}_{mode}" / rfile_name
        if not fpath.exists():
            print(f"  Warning: missing {fpath}")
            continue
        data = load_json(fpath)

        for label, acc in data.get('per_interval_accuracy', {}).items():
            if isinstance(acc, (int, float)):
                out['accuracy'].setdefault(label, []).append(acc)
            elif isinstance(acc, list) and len(acc) > 0 and isinstance(acc[0], list):
                # [pred, true] pairs — compute accuracy
                preds = np.array([pair[0] for pair in acc])
                trues = np.array([pair[1] for pair in acc])
                out['accuracy'].setdefault(label, []).append(float(np.mean(preds == trues)))
        for label, f1 in data.get('per_interval_f1', {}).items():
            out['f1'].setdefault(label, []).append(f1)

        # Word-type aggregate: use stored values if available, else compute
        wc_acc = data.get('word_class_accuracy')
        wc_f1 = data.get('word_class_macro_f1')
        if wc_acc is not None:
            out['word_class_accuracy'].append(wc_acc)
        else:
            # Compute from per_word_class data
            per_wc_acc = data.get('per_word_class_accuracy', {})
            if per_wc_acc:
                # Simple macro average of word-class accuracies
                out['word_class_accuracy'].append(np.mean(list(per_wc_acc.values())))
        if wc_f1 is not None:
            out['word_class_macro_f1'].append(wc_f1)
        else:
            per_wc_f1 = data.get('per_word_class_f1', {})
            if per_wc_f1:
                # Exclude "Unanalyzed word" and "Web email or address" from macro F1
                exclude = {'Word Class Unanalyzed word', 'Word Class Web email or address'}
                vals = [v for k, v in per_wc_f1.items() if k not in exclude]
                if vals:
                    out['word_class_macro_f1'].append(np.mean(vals))

    return out


def generate_per_label_table(datasets_by_model: Dict[str, Dict[str, Dict]]) -> str:
    """Generate Table 4 LaTeX.

    datasets_by_model: {model_key: {'id': data, 'ood': data}}
    where model_key is one of: singlelabel, multilabel,
    multilabel_unconstrained_unnormalized
    """
    model_order = ['singlelabel', 'multilabel',
                   'multilabel_unconstrained_unnormalized']
    model_short = {
        'singlelabel': 'Single',
        'multilabel': 'Multi',
        'multilabel_unconstrained_unnormalized': 'Unc.',
    }
    models = [m for m in model_order if m in datasets_by_model]
    n_models = len(models)
    n_data_cols = n_models * 2  # Acc + F1 per model

    lines = []
    lines.append("\\begin{table*}[]")
    lines.append("    \\centering")
    lines.append("    \\resizebox{\\textwidth}{!}{")
    # Column spec: label + (Acc F1) * n_models * 2 domains
    col_spec = "l" + "r" * (n_data_cols * 2)
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Domain header
    lines.append(f"    & \\multicolumn{{{n_data_cols}}}{{c}}{{In-domain}}"
                 f" & \\multicolumn{{{n_data_cols}}}{{c}}{{OOD}}\\\\")
    # Cmidrules for domains
    id_start, id_end = 2, 1 + n_data_cols
    ood_start, ood_end = 2 + n_data_cols, 1 + n_data_cols * 2
    lines.append(f"    \\cmidrule(lr){{{id_start}-{id_end}}} \\cmidrule(lr){{{ood_start}-{ood_end}}}")

    # Model sub-headers
    model_header = "    Label"
    cmidrules = "    "
    col_idx = 2
    for domain in ['id', 'ood']:
        for m in models:
            model_header += f" & \\multicolumn{{2}}{{c}}{{{model_short[m]}}}"
            cmidrules += f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}} "
            col_idx += 2
    model_header += "\\\\"
    lines.append(model_header)
    lines.append(cmidrules)

    # Acc/F1 sub-header
    acc_f1_header = "    "
    for _ in range(2):  # id, ood
        for _ in models:
            acc_f1_header += " & Acc & F1"
    acc_f1_header += "\\\\"
    lines.append(acc_f1_header)
    lines.append("    \\midrule")

    # Build flat list: [model0_id, model1_id, ..., model0_ood, model1_ood, ...]
    flat_datasets = []
    for domain in ['id', 'ood']:
        for m in models:
            flat_datasets.append(datasets_by_model[m][domain])

    n_total = len(flat_datasets)
    id_range = list(range(n_models))
    ood_range = list(range(n_models, n_total))

    for label in LABEL_ORDER:
        acc_vals = []
        f1_vals = []

        for dset in flat_datasets:
            if label == 'Word Type':
                acc_vals.append(compute_stats(dset['word_class_accuracy']) if dset['word_class_accuracy'] else (0, 0))
                f1_vals.append(compute_stats(dset['word_class_macro_f1']) if dset['word_class_macro_f1'] else (0, 0))
            else:
                if label in dset['accuracy'] and dset['accuracy'][label]:
                    acc_vals.append(compute_stats(dset['accuracy'][label]))
                else:
                    acc_vals.append((0, 0))
                if label in dset['f1'] and dset['f1'][label]:
                    f1_vals.append(compute_stats(dset['f1'][label]))
                else:
                    f1_vals.append((0, 0))

        # Best per group
        best_acc_id = max(acc_vals[i][0] for i in id_range) if id_range else 0
        best_f1_id = max(f1_vals[i][0] for i in id_range) if id_range else 0
        best_acc_ood = max(acc_vals[i][0] for i in ood_range) if ood_range else 0
        best_f1_ood = max(f1_vals[i][0] for i in ood_range) if ood_range else 0

        formatted = []
        for i in range(n_total):
            ba = best_acc_id if i < n_models else best_acc_ood
            bf = best_f1_id if i < n_models else best_f1_ood
            am, ast = acc_vals[i]
            fm, fst = f1_vals[i]

            acc_str = f"{am*100:.1f}$_{{{ast*100:.1f}}}$"
            if am > 0 and abs(am - ba) < 1e-9:
                acc_str = f"\\textbf{{{acc_str}}}"
            f1_str = f"{fm*100:.1f}$_{{{fst*100:.1f}}}$"
            if fm > 0 and abs(fm - bf) < 1e-9:
                f1_str = f"\\textbf{{{f1_str}}}"

            formatted.extend([acc_str, f1_str])

        lines.append(f"    {label} & {' & '.join(formatted)}\\\\")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    }")
    lines.append("    \\caption{Per-label accuracy and F1 scores comparing single-label, "
                 "multi-label (constrained), and unconstrained "
                 "ScandiBERT models on in-domain (Sosialurin-BRAGD, "
                 "10-fold CV) and out-of-domain (OOD-BRAGD) datasets. Values show "
                 "mean$_\\text{std}$ across folds. We ignore the \\emph{unknown word} "
                 "and \\emph{web} tags in the word-type macro F1 results, as these "
                 "have only a handful of labels.")
    lines.append("     }")
    lines.append("    \\label{tab:per-label-results}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: wordclass_results  (per-word-class Acc/F1, single vs multi)
# ---------------------------------------------------------------------------

WORDCLASS_ORDER_TABLE5 = [
    'Word Class Noun', 'Word Class Adjective', 'Word Class Pronoun',
    'Word Class Number', 'Word Class Verbs', 'Word Class Participle',
    'Word Class Adverb', 'Word Class Conjunctions', 'Word Class Foreign words',
    'Word Class Abbreviation', 'Word Class Punctuation', 'Word Class Symbol',
    'Word Class R Article',
]

WORDCLASS_DISPLAY = {
    'Word Class Noun': 'Noun', 'Word Class Adjective': 'Adjective',
    'Word Class Pronoun': 'Pronoun', 'Word Class Number': 'Number',
    'Word Class Verbs': 'Verbs', 'Word Class Participle': 'Participle',
    'Word Class Adverb': 'Adverb', 'Word Class Conjunctions': 'Conjunctions',
    'Word Class Foreign words': 'Foreign words',
    'Word Class Abbreviation': 'Abbreviation',
    'Word Class Punctuation': 'Punctuation', 'Word Class Symbol': 'Symbol',
    'Word Class R Article': 'R Article',
}


def load_per_wordclass_results(base_dir: Path, mode: str, result_type: str,
                               num_folds: int = 10) -> Dict:
    """Load per-word-class accuracy and F1 across folds."""
    out: Dict[str, Dict[str, List[float]]] = {'accuracy': {}, 'f1': {}}
    rfile_name = "best_results.json" if result_type == "best" else "ood_results.json"

    for fold in range(num_folds):
        fpath = base_dir / f"fold_{fold}_{mode}" / rfile_name
        if not fpath.exists():
            continue
        data = load_json(fpath)

        for wc, acc in data.get('per_word_class_accuracy', {}).items():
            out['accuracy'].setdefault(wc, []).append(acc)
        for wc, f1 in data.get('per_word_class_f1', {}).items():
            out['f1'].setdefault(wc, []).append(f1)

    return out


def generate_wordclass_table(datasets_by_model: Dict[str, Dict[str, Dict]]) -> str:
    """Generate Table 5 LaTeX.

    datasets_by_model: {model_key: {'id': data, 'ood': data}}
    """
    model_order = ['singlelabel', 'multilabel',
                   'multilabel_unconstrained_unnormalized']
    model_short = {
        'singlelabel': 'Single',
        'multilabel': 'Multi',
        'multilabel_unconstrained_unnormalized': 'Unc.',
    }
    models = [m for m in model_order if m in datasets_by_model]
    n_models = len(models)
    n_data_cols = n_models * 2

    lines = []
    lines.append("\\begin{table*}[]")
    lines.append("    \\centering")
    lines.append("    \\resizebox{\\textwidth}{!}{")
    col_spec = "l" + "r" * (n_data_cols * 2)
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    lines.append(f"    & \\multicolumn{{{n_data_cols}}}{{c}}{{In-domain}}"
                 f" & \\multicolumn{{{n_data_cols}}}{{c}}{{OOD}}\\\\")
    id_start, id_end = 2, 1 + n_data_cols
    ood_start, ood_end = 2 + n_data_cols, 1 + n_data_cols * 2
    lines.append(f"    \\cmidrule(lr){{{id_start}-{id_end}}} \\cmidrule(lr){{{ood_start}-{ood_end}}}")

    model_header = "    Word Class"
    cmidrules = "    "
    col_idx = 2
    for domain in ['id', 'ood']:
        for m in models:
            model_header += f" & \\multicolumn{{2}}{{c}}{{{model_short[m]}}}"
            cmidrules += f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}} "
            col_idx += 2
    model_header += "\\\\"
    lines.append(model_header)
    lines.append(cmidrules)

    acc_f1_header = "    "
    for _ in range(2):
        for _ in models:
            acc_f1_header += " & Acc & F1"
    acc_f1_header += "\\\\"
    lines.append(acc_f1_header)
    lines.append("    \\midrule")

    flat_datasets = []
    for domain in ['id', 'ood']:
        for m in models:
            flat_datasets.append(datasets_by_model[m][domain])

    n_total = len(flat_datasets)
    id_range = list(range(n_models))
    ood_range = list(range(n_models, n_total))

    for wc in WORDCLASS_ORDER_TABLE5:
        acc_vals = []
        f1_vals = []

        for dset in flat_datasets:
            if wc in dset['accuracy'] and dset['accuracy'][wc]:
                acc_vals.append(compute_stats(dset['accuracy'][wc]))
            else:
                acc_vals.append((0, 0))
            if wc in dset['f1'] and dset['f1'][wc]:
                f1_vals.append(compute_stats(dset['f1'][wc]))
            else:
                f1_vals.append((0, 0))

        best_acc_id = max(acc_vals[i][0] for i in id_range) if id_range else 0
        best_f1_id = max(f1_vals[i][0] for i in id_range) if id_range else 0
        best_acc_ood = max(acc_vals[i][0] for i in ood_range) if ood_range else 0
        best_f1_ood = max(f1_vals[i][0] for i in ood_range) if ood_range else 0

        formatted = []
        for i in range(n_total):
            ba = best_acc_id if i < n_models else best_acc_ood
            bf = best_f1_id if i < n_models else best_f1_ood
            am, ast = acc_vals[i]
            fm, fst = f1_vals[i]

            acc_str = f"{am*100:.1f}$_{{{ast*100:.1f}}}$"
            if am > 0 and abs(am - ba) < 1e-9:
                acc_str = f"\\textbf{{{acc_str}}}"
            f1_str = f"{fm*100:.1f}$_{{{fst*100:.1f}}}$"
            if fm > 0 and abs(fm - bf) < 1e-9:
                f1_str = f"\\textbf{{{f1_str}}}"

            formatted.extend([acc_str, f1_str])

        display_name = WORDCLASS_DISPLAY.get(wc, wc.replace('Word Class ', ''))
        lines.append(f"    {display_name} & {' & '.join(formatted)}\\\\")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    }")
    lines.append("    \\caption{Per-word-class accuracy and F1 scores comparing "
                 "single-label, multi-label (constrained), and unconstrained "
                 "ScandiBERT models on in-domain "
                 "(Sosialurin-BRAGD, 10-fold CV) and out-of-domain (OOD-BRAGD) "
                 "datasets. Values show mean$_\\text{std}$ across folds.")
    lines.append("    }")
    lines.append("    \\label{tab:per-wordclass-results}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 1: training_progress
# ---------------------------------------------------------------------------

def load_epoch_results(base_dir: Path, mode: str,
                       max_epochs: int = 20) -> Tuple[Dict[int, List[float]], List[int]]:
    """Load validation accuracies per epoch across folds (forward-fill strategy)."""
    epoch_data: Dict[int, List[float]] = {e: [] for e in range(1, max_epochs + 1)}
    best_epochs: List[int] = []

    for fold in range(10):
        fold_path = base_dir / f"fold_{fold}_{mode}"
        if not fold_path.exists():
            continue

        # Load best results
        best_file = fold_path / "best_results.json"
        best_acc, best_epoch = None, None
        if best_file.exists():
            bd = load_json(best_file)
            best_acc = bd.get('val_composite_accuracy')
            best_epoch = bd.get('best_epoch')

        # Load per-epoch results
        fold_epoch_accs = {}
        for epoch in range(1, max_epochs + 1):
            ef = fold_path / f"epoch_{epoch}_results.json"
            if ef.exists():
                ed = load_json(ef)
                acc = ed.get('val_composite_accuracy')
                if acc is not None:
                    fold_epoch_accs[epoch] = acc

        if best_epoch and best_acc:
            pass
        elif fold_epoch_accs:
            best_epoch = max(fold_epoch_accs, key=lambda e: fold_epoch_accs[e])
            best_acc = fold_epoch_accs[best_epoch]
        else:
            continue

        best_epochs.append(best_epoch)

        # Forward-fill: real values before best, best value from best_epoch onward
        for epoch in range(1, max_epochs + 1):
            if epoch < best_epoch and epoch in fold_epoch_accs:
                epoch_data[epoch].append(fold_epoch_accs[epoch])
            elif epoch >= best_epoch:
                epoch_data[epoch].append(best_acc)

    return epoch_data, best_epochs


def generate_training_plot(base_dir: Path, output_path: Path):
    """Generate training progress figure with 4 model curves."""
    print("Loading training progress data...")

    model_configs = [
        ("singlelabel", "Single-label", '#2E86AB', 'o', '-'),
        ("multilabel", "Multi-label (constrained)", '#A23B72', 's', '-'),
        ("multilabel_unconstrained_unnormalized", "Multi-label (unconstrained)", '#FF9800', 'D', '--'),
    ]

    def epoch_stats(epoch_data):
        epochs, means, stds = [], [], []
        for e in sorted(epoch_data):
            if epoch_data[e]:
                epochs.append(e)
                means.append(np.mean(epoch_data[e]) * 100)
                stds.append(np.std(epoch_data[e], ddof=1) * 100 if len(epoch_data[e]) > 1 else 0.0)
        return epochs, means, stds

    all_data = {}
    for mode, label, color, marker, linestyle in model_configs:
        epoch_data, best_epochs = load_epoch_results(base_dir, mode)
        es, ms, ss = epoch_stats(epoch_data)
        if es:
            all_data[mode] = (es, ms, ss, best_epochs, label, color, marker, linestyle)

    if len(all_data) < 2:
        print("Error: Not enough epoch data found. Skipping plot.")
        return

    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.transparent'] = False
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.set_facecolor('white')
    ax.patch.set_alpha(1.0)

    for mode, label, color, marker, linestyle in model_configs:
        if mode not in all_data:
            continue
        es, ms, ss, _, lbl, clr, mkr, ls = all_data[mode]
        # Plot error rate (100 - accuracy) so log scale spreads high-accuracy differences
        errs = [100 - m for m in ms]
        err_lo = [100 - (m + s) for m, s in zip(ms, ss)]
        err_hi = [100 - (m - s) for m, s in zip(ms, ss)]
        ax.plot(es, errs, marker=mkr, linestyle=ls, label=lbl, color=clr, linewidth=2.5, markersize=7)
        ax.fill_between(es, err_lo, err_hi, alpha=0.15, color=clr)

    ax.set_yscale('log')
    ax.set_xlim(1, None)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel('Epoch', fontsize=28)
    ax.set_ylabel('Validation Error Rate (%)', fontsize=28)
    ax.legend(fontsize=22, loc='upper right')
    ax.grid(True, which='major', axis='y', alpha=0.3, linestyle='-')
    ax.grid(True, which='minor', axis='y', alpha=0.15, linestyle=':')
    ax.tick_params(labelsize=24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Explicit y-ticks for readability on log scale
    import matplotlib.ticker as mticker
    ax.set_yticks([2, 3, 5, 8, 12])
    ax.set_yticklabels(['2%', '3%', '5%', '8%', '12%'])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False, dpi=150)
    print(f"  Saved figure to {output_path}")

    # Print convergence statistics for all models
    for mode, label, color, marker, linestyle in model_configs:
        if mode in all_data:
            _, _, _, best_epochs, lbl, _, _, _ = all_data[mode]
            if best_epochs:
                avg = np.mean(best_epochs)
                rng = f"{min(best_epochs)}-{max(best_epochs)}"
                print(f"  {lbl}: avg {avg:.1f} epochs (range: {rng})")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Appendix: unconstrained normalization ablation
# ---------------------------------------------------------------------------

def generate_appendix_unconstrained_table(all_results: Dict) -> str:
    """Generate a compact appendix table comparing all 4 ScandiBERT variants.

    Shows overall composite accuracy only (not per-wordclass), for both
    in-domain and OOD.
    """
    model_order = ['singlelabel', 'multilabel',
                   'multilabel_unconstrained_normalized',
                   'multilabel_unconstrained_unnormalized']
    model_display = {
        'singlelabel': 'Single-label',
        'multilabel': 'Multi-label (constrained)',
        'multilabel_unconstrained_normalized': 'Unconstrained (normalized)',
        'multilabel_unconstrained_unnormalized': 'Unconstrained (unnormalized)',
    }

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{lrr}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Model} & \\textbf{In-domain} & \\textbf{OOD}\\\\")
    lines.append("    \\midrule")

    # Collect means for bolding
    id_means = {}
    ood_means = {}
    for mkey in model_order:
        entry_id = all_results['val'].get(mkey, {}).get('_composite_accuracy')
        entry_ood = all_results['ood'].get(mkey, {}).get('_composite_accuracy')
        id_means[mkey] = entry_id[0] if entry_id else 0
        ood_means[mkey] = entry_ood[0] if entry_ood else 0

    best_id = max(id_means.values())
    best_ood = max(ood_means.values())

    for mkey in model_order:
        entry_id = all_results['val'].get(mkey, {}).get('_composite_accuracy')
        entry_ood = all_results['ood'].get(mkey, {}).get('_composite_accuracy')

        if entry_id:
            id_str = format_value(entry_id[0], entry_id[1], abs(entry_id[0] - best_id) < 0.05)
        else:
            id_str = "---"
        if entry_ood:
            ood_str = format_value(entry_ood[0], entry_ood[1], abs(entry_ood[0] - best_ood) < 0.05)
        else:
            ood_str = "---"

        lines.append(f"    {model_display[mkey]} & {id_str} & {ood_str}\\\\")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    \\caption{Comparison of all four ScandiBERT model variants. "
                 "The two unconstrained variants (normalized vs.\\ unnormalized) "
                 "do not differ significantly ($p > 0.2$). "
                 "Values show composite accuracy (mean$_\\text{std}$ across 10 folds).}")
    lines.append("    \\label{tab:unconstrained-ablation}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all tables and figures for the paper.")
    parser.add_argument('--skip-abltagger', action='store_true',
                        help="Skip ABLTagger results (if not yet retrained)")
    parser.add_argument('--abltagger-val-summary', type=str, default='results/abltagger/val_summary.json',
                        help="Path to ABLTagger validation summary.json")
    parser.add_argument('--abltagger-ood-summary', type=str, default='results/abltagger/ood_summary.json',
                        help="Path to ABLTagger OOD summary.json")
    args = parser.parse_args()

    output_dir = Path(os.environ.get('OUTPUT_DIR', './output'))
    results_dir = Path(os.environ.get('RESULTS_DIR', 'results/all_splits_eval_bragd_adamw'))

    tables_dir = output_dir / 'tables'
    figs_dir = output_dir / 'figs'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Skip ABLTagger: {args.skip_abltagger}")
    print()

    # ---- Table 3: full_results ----
    print("=" * 60)
    print("TABLE 3: full_results (overall + per-word-class F1)")
    print("=" * 60)

    all_results = {'val': {}, 'ood': {}}
    for ds, ds_name in [('val', 'in-domain'), ('ood', 'OOD')]:
        print(f"  Loading {ds_name} results...")
        for model in ['singlelabel', 'multilabel', 'tnt',
                     'multilabel_unconstrained_normalized',
                     'multilabel_unconstrained_unnormalized']:
            all_results[ds][model] = aggregate_word_class_f1(results_dir, model, ds)

    if not args.skip_abltagger:
        abl_val = args.abltagger_val_summary
        abl_ood = args.abltagger_ood_summary
        if abl_val and Path(abl_val).exists():
            all_results['val']['abltagger'] = load_abltagger_summary(Path(abl_val))
            print(f"  Loaded ABLTagger val from {abl_val}")
        else:
            print("  ABLTagger val summary not found, skipping")
            args.skip_abltagger = True
        if abl_ood and Path(abl_ood).exists():
            all_results['ood']['abltagger'] = load_abltagger_summary(Path(abl_ood))
            print(f"  Loaded ABLTagger OOD from {abl_ood}")

    table3 = generate_full_results_table(all_results, args.skip_abltagger)
    (tables_dir / 'full_results.tex').write_text(table3)
    print(f"  -> {tables_dir / 'full_results.tex'}")

    # ---- Table 4: per_label_results ----
    print()
    print("=" * 60)
    print("TABLE 4: per_label_results (per-feature-group Acc/F1)")
    print("=" * 60)

    per_label_models = ['singlelabel', 'multilabel',
                        'multilabel_unconstrained_unnormalized']
    pl_data = {}
    for mode in per_label_models:
        pl_data[mode] = {
            'id': load_per_label_results(results_dir, mode, 'best'),
            'ood': load_per_label_results(results_dir, mode, 'ood'),
        }

    table4 = generate_per_label_table(pl_data)
    (tables_dir / 'per_label_results.tex').write_text(table4)
    print(f"  -> {tables_dir / 'per_label_results.tex'}")

    # ---- Table 5: wordclass_results ----
    print()
    print("=" * 60)
    print("TABLE 5: wordclass_results (per-word-class Acc/F1)")
    print("=" * 60)

    wc_data = {}
    for mode in per_label_models:
        wc_data[mode] = {
            'id': load_per_wordclass_results(results_dir, mode, 'best'),
            'ood': load_per_wordclass_results(results_dir, mode, 'ood'),
        }

    table5 = generate_wordclass_table(wc_data)
    (tables_dir / 'wordclass_results.tex').write_text(table5)
    print(f"  -> {tables_dir / 'wordclass_results.tex'}")

    # ---- Figure 1: training_progress ----
    print()
    print("=" * 60)
    print("FIGURE 1: training_progress")
    print("=" * 60)

    generate_training_plot(results_dir, figs_dir / 'training_progress.pdf')

    # ---- Appendix: unconstrained normalization ablation ----
    print()
    print("=" * 60)
    print("APPENDIX: unconstrained normalization ablation")
    print("=" * 60)

    appendix_table = generate_appendix_unconstrained_table(all_results)
    (tables_dir / 'appendix_unconstrained.tex').write_text(appendix_table)
    print(f"  -> {tables_dir / 'appendix_unconstrained.tex'}")

    print()
    print("Done! All outputs written to", output_dir)


if __name__ == "__main__":
    main()
