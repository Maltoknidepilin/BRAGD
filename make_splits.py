# make_splits.py
# Creates split_indices.npy based on the CURRENT version of your corpus TSV.
# Run from the repo root:
#   python .\make_splits.py
#
# If you edit EOS\tEOS boundaries or corpus formatting, rerun this to regenerate splits.

import os
import numpy as np
import pandas as pd


# =========================
# CHANGE THESE IF NEEDED
# =========================

TAGS_FILEPATH = "data/Sosialurin-BRAGD_tags.csv"   # <-- change if your tags CSV path changes
CORPUS_FILEPATH = "data/Sosialurin-BRAGD.tsv"      # <-- change if your corpus TSV path changes
OUTPUT_SPLITS = "split_indices.npy"                # <-- where to save the splits file

NUM_FOLDS = 10                                     # <-- change to 5 if you want 5-fold CV, etc.
RANDOM_SEED = 42                                   # <-- change if you want different random splits

# Behavior switches:
SKIP_HEADER = True                                 # <-- keep True if TSV has token<TAB>tag header line
KEEP_EMPTY_SENTENCES = False                        # <-- matches your training loader: False means drop empty
WARN_LIMIT = 30                                     # <-- max examples printed for bad lines / unknown tags


def load_tag_to_features(tags_filepath: str):
    """
    Reads the tags CSV and builds a mapping:
      Original Tag -> feature vector (numpy int array)
    """
    tags_df = pd.read_csv(tags_filepath)
    if "Original Tag" not in tags_df.columns:
        raise ValueError("Tags CSV must contain a column named 'Original Tag'")

    tag_to_features = {
        row["Original Tag"]: row[1:].values.astype(int)
        for _, row in tags_df.iterrows()
    }
    return tag_to_features


def build_sentences_from_tsv(corpus_filepath: str, tag_to_features: dict):
    """
    Builds the sentence list exactly like your training loader:
    - sentence boundary: a line equal to 'EOS\\tEOS'
    - token lines must be exactly: TOKEN\\tTAG (2 columns)
    - if TAG not in tag_to_features, that token is skipped
    - sentence is appended only if non-empty (unless KEEP_EMPTY_SENTENCES=True)
    - header 'token\\ttag' is skipped if SKIP_HEADER=True
    """
    sentences = []
    current_sentence = []

    eos_count = 0
    header_skipped = 0
    bad_format = 0
    unknown_tags = 0
    kept_tokens = 0

    def maybe_append_sentence():
        nonlocal sentences, current_sentence
        if current_sentence or KEEP_EMPTY_SENTENCES:
            if current_sentence:  # only count real sentences as having tokens
                sentences.append(current_sentence)
            else:
                sentences.append([])
        current_sentence = []

    with open(corpus_filepath, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # Optional header skip
            if SKIP_HEADER and line.lower().strip() in ("token\ttag", "word\ttag"):
                header_skipped += 1
                continue

            # Sentence boundary
            if line.strip() == "EOS\tEOS":
                eos_count += 1
                maybe_append_sentence()
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                bad_format += 1
                if bad_format <= WARN_LIMIT:
                    print(f"[BAD FORMAT] line {lineno}: {line}")
                continue

            token, tag = parts
            if tag not in tag_to_features:
                unknown_tags += 1
                if unknown_tags <= WARN_LIMIT:
                    print(f"[UNKNOWN TAG] line {lineno}: token={token!r} tag={tag!r}")
                continue

            current_sentence.append(token)
            kept_tokens += 1

    # Handle file not ending with EOS\tEOS
    if current_sentence or KEEP_EMPTY_SENTENCES:
        maybe_append_sentence()

    print("\n--- Corpus summary ---")
    print(f"EOS\\tEOS lines: {eos_count}")
    print(f"Header lines skipped: {header_skipped}")
    print(f"Bad format lines (not 2 columns): {bad_format}")
    print(f"Unknown tags skipped: {unknown_tags}")
    print(f"Kept tokens: {kept_tokens}")
    print(f"Number of sentences (used for splits): {len(sentences)}")

    return sentences


def make_kfold_splits(n_sentences: int, k: int, seed: int):
    """
    Makes K folds by shuffling indices with a fixed seed,
    then splitting into k roughly equal chunks.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_sentences)
    folds = np.array_split(idx, k)

    splits = {}
    for fold in range(k):
        val_idx = np.sort(folds[fold]).astype(int).tolist()
        train_idx = np.sort(np.concatenate([folds[i] for i in range(k) if i != fold])).astype(int).tolist()
        splits[fold] = {"train": train_idx, "val": val_idx}

    return splits


def main():
    # Basic file checks
    if not os.path.exists(TAGS_FILEPATH):
        raise FileNotFoundError(f"Tags file not found: {TAGS_FILEPATH}")
    if not os.path.exists(CORPUS_FILEPATH):
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_FILEPATH}")

    tag_to_features = load_tag_to_features(TAGS_FILEPATH)
    sentences = build_sentences_from_tsv(CORPUS_FILEPATH, tag_to_features)
    n = len(sentences)

    if n < NUM_FOLDS:
        raise ValueError(f"Not enough sentences ({n}) for {NUM_FOLDS} folds.")

    splits = make_kfold_splits(n_sentences=n, k=NUM_FOLDS, seed=RANDOM_SEED)
    np.save(OUTPUT_SPLITS, splits)

    # Sanity check: max index must be <= n-1
    mx = max(max(splits[f]["train"] + splits[f]["val"]) for f in splits)
    print("\n--- Splits saved ---")
    print(f"Output file: {OUTPUT_SPLITS}")
    print(f"Folds: {NUM_FOLDS}")
    print(f"Sentences: {n}")
    print(f"Max index in splits: {mx} (should be <= {n-1})")


if __name__ == "__main__":
    main()