"""
Inference script for the Faroese POS tagger.

Usage as CLI:
    python inference.py "Hetta er eitt domi"
    echo "Hetta er eitt domi" | python inference.py

Usage as library:
    from inference import load_tagger
    tagger = load_tagger("Setur/BRAGD")
    results = tagger.tag("Hetta er eitt domi")
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification


# ---------------------------------------------------------------------------
# Feature layout (73 binary features)
# ---------------------------------------------------------------------------

INTERVALS = (
    (15, 29),   # Subcategories
    (30, 33),   # Gender
    (34, 36),   # Number
    (37, 41),   # Case
    (42, 43),   # Article
    (44, 45),   # Proper Noun
    (46, 50),   # Degree
    (51, 53),   # Declension
    (54, 60),   # Mood
    (61, 63),   # Voice
    (64, 66),   # Tense
    (67, 70),   # Person
    (71, 72),   # Definiteness
)

INTERVAL_NAMES = {
    (15, 29): "subcategory",
    (30, 33): "gender",
    (34, 36): "number",
    (37, 41): "case",
    (42, 43): "article",
    (44, 45): "proper_noun",
    (46, 50): "degree",
    (51, 53): "declension",
    (54, 60): "mood",
    (61, 63): "voice",
    (64, 66): "tense",
    (67, 70): "person",
    (71, 72): "definiteness",
}

WORD_CLASS_NAMES = {
    0: "Noun",
    1: "Adjective",
    2: "Pronoun",
    3: "Number",
    4: "Verb",
    5: "Participle",
    6: "Adverb",
    7: "Conjunction",
    8: "Foreign",
    9: "Unanalyzed",
    10: "Abbreviation",
    11: "Web",
    12: "Punctuation",
    13: "Symbol",
    14: "Article",
}

# Column headers from the tags CSV (indices 1..73, after "Original Tag")
# These give the short label for each binary feature position.
FEATURE_COLUMNS = [
    # 0-14: Word classes
    "S", "A", "P", "N", "V", "L", "D", "C", "F", "X", "T", "W", "K", "M", "R",
    # 15-29: Subcategories
    "D", "B", "E", "I", "P", "Q", "N", "G", "R", "X", "S", "C", "O", "T", "s",
    # 30-33: Gender
    "M", "F", "N", "g",
    # 34-36: Number
    "S", "P", "n",
    # 37-41: Case
    "N", "A", "D", "G", "c",
    # 42-43: Article
    "A", "a",
    # 44-45: Proper Noun
    "P", "r",
    # 46-50: Degree
    "P", "C", "S", "A", "d",
    # 51-53: Declension
    "S", "W", "e",
    # 54-60: Mood
    "I", "M", "N", "S", "P", "E", "U",
    # 61-63: Voice
    "A", "M", "v",
    # 64-66: Tense
    "P", "A", "t",
    # 67-70: Person
    "1", "2", "3", "p",
    # 71-72: Definiteness
    "D", "I",
]


# ---------------------------------------------------------------------------
# Constraint mask helpers
# ---------------------------------------------------------------------------

def _build_constraint_mask_from_csv(tags_csv_path: str) -> dict[int, list[tuple[int, int]]]:
    """Build the constraint mask (dict_intervals) from the tags CSV.

    For each of the 15 word classes, determine which feature-group intervals
    are active (i.e. have at least one non-zero value across all tags of that
    word class).

    Returns:
        dict mapping word-class index (0-14) to list of (start, end) tuples.
    """
    import pandas as pd
    tags_df = pd.read_csv(tags_csv_path)
    tag_to_features = {
        row["Original Tag"]: row.iloc[1:].values.astype(int)
        for _, row in tags_df.iterrows()
    }

    unique_arrays = list({tuple(v) for v in tag_to_features.values()})
    unique_arrays = [np.array(t) for t in unique_arrays]

    dict_intervals: dict[int, list[tuple[int, int]]] = {}
    for wc in range(15):
        wc_arrays = np.array([a for a in unique_arrays if a[wc] == 1])
        if len(wc_arrays) == 0:
            dict_intervals[wc] = []
            continue
        col_sums = wc_arrays.sum(axis=0)
        allowed = [
            iv for iv in INTERVALS
            if col_sums[iv[0]: iv[1] + 1].sum() > 0
        ]
        dict_intervals[wc] = allowed
    return dict_intervals


def _build_features_to_tag_from_csv(tags_csv_path: str) -> dict[tuple, str]:
    """Build mapping from feature-vector tuples to composite tag strings."""
    import pandas as pd
    tags_df = pd.read_csv(tags_csv_path)
    return {
        tuple(row.iloc[1:].values.astype(int)): row["Original Tag"]
        for _, row in tags_df.iterrows()
    }


# ---------------------------------------------------------------------------
# Tagger class
# ---------------------------------------------------------------------------

class Tagger:
    """Faroese POS tagger using a fine-tuned XLM-R model with constrained
    multi-label decoding."""

    def __init__(
        self,
        model: XLMRobertaForTokenClassification,
        tokenizer: XLMRobertaTokenizerFast,
        device: torch.device,
        constraint_mask: dict[int, list[tuple[int, int]]],
        features_to_tag: dict[tuple, str],
        max_len: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.constraint_mask = constraint_mask
        self.features_to_tag = features_to_tag
        self.max_len = max_len
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, text: str) -> list[dict]:
        """Tag a raw text string.

        The text is split on whitespace into tokens.  Returns a list of dicts,
        one per token, with keys ``word``, ``tag``, and ``features``.
        """
        words = text.split()
        if not words:
            return []
        results = self.tag_sentences([words])
        return results[0]

    def tag_sentences(self, sentences: list[list[str]]) -> list[list[dict]]:
        """Tag a batch of pre-tokenized sentences.

        All sentences are padded and passed through the model in a single
        forward pass for efficient GPU utilisation.

        Args:
            sentences: List of sentences, each a list of word strings.

        Returns:
            List of sentences, each a list of dicts with keys
            ``word``, ``tag``, and ``features``.
        """
        if not sentences:
            return []

        # Tokenize all sentences, collecting begin-token masks per sentence
        encodings = self.tokenizer(
            sentences,
            is_split_into_words=True,
            padding="longest",
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        begin_tokens: list[list[int]] = []
        for batch_idx in range(len(sentences)):
            word_ids = encodings.word_ids(batch_index=batch_idx)
            bt = []
            prev_word_id = None
            for wid in word_ids:
                if wid is not None and wid != prev_word_id:
                    bt.append(1)
                else:
                    bt.append(0)
                prev_word_id = wid
            begin_tokens.append(bt)

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits = outputs.logits  # (batch, seq_len, num_features)

        all_results: list[list[dict]] = []
        for idx, words in enumerate(sentences):
            predictions = self._decode_sequence(
                all_logits[idx], attention_mask[idx], begin_tokens[idx]
            )

            sentence_results = []
            word_idx = 0
            for pred_vec in predictions:
                tag_str = self._vector_to_tag(pred_vec)
                features = self._vector_to_features(pred_vec)
                sentence_results.append({
                    "word": words[word_idx] if word_idx < len(words) else "?",
                    "tag": tag_str,
                    "features": features,
                })
                word_idx += 1

            all_results.append(sentence_results)

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_sequence(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        begin_token: list[int],
    ) -> list[np.ndarray]:
        """Greedy constrained decoding for one sequence."""
        softmax = torch.nn.Softmax(dim=0)
        predictions = []

        for i in range(len(attention_mask)):
            if attention_mask[i] == 1 and begin_token[i] == 1 and i > 0:
                pred_logits = logits[i]
                num_features = pred_logits.shape[0]
                prediction = torch.zeros(num_features, device=self.device)

                # Step 1: predict word class (features 0-14)
                wc_logits = pred_logits[:15]
                wc_probs = softmax(wc_logits)
                wc_idx = torch.argmax(wc_probs).item()
                prediction[wc_idx] = 1

                # Step 2: decode only valid feature groups for this word class
                for start, end in self.constraint_mask.get(wc_idx, []):
                    group_logits = pred_logits[start: end + 1]
                    group_probs = softmax(group_logits)
                    best = torch.argmax(group_probs).item()
                    prediction[start + best] = 1

                predictions.append(prediction.cpu().numpy().astype(int))

        return predictions

    def _vector_to_tag(self, vec: np.ndarray) -> str:
        """Convert a binary feature vector to its composite tag string."""
        tag = self.features_to_tag.get(tuple(vec))
        if tag is not None:
            return tag
        # Fallback: build a tag string from the active feature labels
        parts = []
        for i, v in enumerate(vec):
            if v == 1:
                parts.append(FEATURE_COLUMNS[i])
        return "".join(parts)

    @staticmethod
    def _vector_to_features(vec: np.ndarray) -> dict:
        """Convert a binary feature vector to a human-readable feature dict."""
        features: dict = {}

        # Word class
        wc_idx = None
        for i in range(15):
            if vec[i] == 1:
                wc_idx = i
                features["word_class"] = WORD_CLASS_NAMES.get(i, str(i))
                break

        # Feature groups
        for (start, end), name in INTERVAL_NAMES.items():
            group = vec[start: end + 1]
            active = np.where(group == 1)[0]
            if len(active) == 1:
                offset = active[0]
                label = FEATURE_COLUMNS[start + offset]
                features[name] = label
            elif len(active) > 1:
                labels = [FEATURE_COLUMNS[start + o] for o in active]
                features[name] = ",".join(labels)
            # If no feature is active in this group, omit it.

        return features


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tagger(
    model_name_or_path: str = "Setur/BRAGD",
    device: Optional[str] = None,
    tags_csv_fallback: Optional[str] = None,
) -> Tagger:
    """Load a Faroese POS tagger.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
        device: Device string (e.g. ``"cuda"`` or ``"cpu"``).  Auto-detected
            if *None*.
        tags_csv_fallback: Path to ``Sosialurin-BRAGD_tags.csv`` used when
            ``constraint_mask.json`` / ``tag_mappings.json`` are not found
            alongside the model.  Defaults to ``data/Sosialurin-BRAGD_tags.csv``
            relative to this script.

    Returns:
        A :class:`Tagger` instance ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model and tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name_or_path)
    model = XLMRobertaForTokenClassification.from_pretrained(model_name_or_path)
    model.to(device)

    # Try to load pre-computed JSON artifacts from the model directory
    constraint_mask = None
    features_to_tag = None

    # For HF hub models, try downloading the JSON files
    model_dir = model_name_or_path
    if not os.path.isdir(model_dir):
        # model_name_or_path is a HF hub ID; try to fetch JSONs via hf_hub
        try:
            from huggingface_hub import hf_hub_download
            cm_path = hf_hub_download(model_name_or_path, "constraint_mask.json")
            with open(cm_path) as f:
                raw = json.load(f)
            constraint_mask = {
                int(k): [tuple(iv) for iv in v] for k, v in raw.items()
            }
        except Exception:
            constraint_mask = None

        try:
            from huggingface_hub import hf_hub_download
            tm_path = hf_hub_download(model_name_or_path, "tag_mappings.json")
            with open(tm_path) as f:
                raw = json.load(f)
            features_to_tag = {tuple(map(int, k.split(","))): v for k, v in raw.items()}
        except Exception:
            features_to_tag = None
    else:
        # Local directory
        cm_file = os.path.join(model_dir, "constraint_mask.json")
        if os.path.isfile(cm_file):
            with open(cm_file) as f:
                raw = json.load(f)
            constraint_mask = {
                int(k): [tuple(iv) for iv in v] for k, v in raw.items()
            }

        tm_file = os.path.join(model_dir, "tag_mappings.json")
        if os.path.isfile(tm_file):
            with open(tm_file) as f:
                raw = json.load(f)
            features_to_tag = {tuple(map(int, k.split(","))): v for k, v in raw.items()}

    # Fallback: build from CSV
    if constraint_mask is None or features_to_tag is None:
        if tags_csv_fallback is None:
            script_dir = Path(__file__).resolve().parent
            tags_csv_fallback = str(script_dir / "data" / "Sosialurin-BRAGD_tags.csv")

        if not os.path.isfile(tags_csv_fallback):
            raise FileNotFoundError(
                f"Cannot find tag metadata. Looked for constraint_mask.json / "
                f"tag_mappings.json in model directory, and fallback CSV at "
                f"{tags_csv_fallback}"
            )

        print(f"[inference] Building tag metadata from {tags_csv_fallback}")
        if constraint_mask is None:
            constraint_mask = _build_constraint_mask_from_csv(tags_csv_fallback)
        if features_to_tag is None:
            features_to_tag = _build_features_to_tag_from_csv(tags_csv_fallback)

    return Tagger(
        model=model,
        tokenizer=tokenizer,
        device=device,
        constraint_mask=constraint_mask,
        features_to_tag=features_to_tag,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_results(results: list[dict]) -> str:
    """Format tagged results as a readable table."""
    if not results:
        return "(no tokens)"

    # Find max widths
    max_word = max(len(r["word"]) for r in results)
    max_tag = max(len(r["tag"]) for r in results)

    lines = []
    for r in results:
        feat_str = ", ".join(
            f"{k}={v}" for k, v in r["features"].items()
        )
        lines.append(
            f"  {r['word']:<{max_word}}  {r['tag']:<{max_tag}}  {feat_str}"
        )
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Faroese POS tagger inference",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to tag (reads from stdin if omitted)",
    )
    parser.add_argument(
        "--model",
        default="Setur/BRAGD",
        help="HuggingFace model ID or local checkpoint path "
             "(default: Setur/BRAGD)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tags-csv",
        default=None,
        help="Path to Sosialurin-BRAGD_tags.csv (fallback if JSONs missing)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Determine input text
    if args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    if not text:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    tagger = load_tagger(
        model_name_or_path=args.model,
        device=args.device,
        tags_csv_fallback=args.tags_csv,
    )

    results = tagger.tag(text)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(_format_results(results))


if __name__ == "__main__":
    main()
