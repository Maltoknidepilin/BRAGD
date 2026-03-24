try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, Adafactor, get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
import evaluate
import argparse
import random
import os
import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

# Import data utilities
from data_utils import (
    load_and_process_corpus,
    load_ood_data,
    create_tag_mappings,
    prepare_tnt_data,
    load_split_indices,
    train_test_split_data,
    vector_to_composite_tag
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Command-line arguments
parser = argparse.ArgumentParser(description='Train a POS tagger.')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer (ignored if using Adafactor with relative_step=True)')
parser.add_argument('--fold', type=int, default=0, help='Cross validation fold')
parser.add_argument('--mode', type=str, default='multilabel', choices=['multilabel', 'singlelabel'], help='Training mode: multilabel or singlelabel')
parser.add_argument('--model_type', type=str, default='neural', choices=['neural', 'tnt'], help='Model type: neural (transformer) or tnt (statistical baseline)')
parser.add_argument('--optimizer', type=str, default='adafactor', choices=['adafactor', 'adam', 'adamw'], help='Optimizer: adafactor (adaptive lr), adam (fixed lr), or adamw (fixed lr with weight decay)')
parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio for learning rate scheduler (only used with adam/adamw, mutually exclusive with warmup_steps)')
parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps for learning rate scheduler (only used with adam/adamw, mutually exclusive with warmup_ratio)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation (default: 64, larger than training for faster inference)')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='Base directory for output checkpoints and results')
parser.add_argument('--evaluate_ood', action='store_true', help='Evaluate on OOD test set after training')
parser.add_argument('--ood_data_path', type=str, default='data/OOD-BRAGD.json', help='Path to OOD JSON data')
parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate (skip training), requires --checkpoint_path')
parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint for evaluation-only mode')
parser.add_argument('--eval_decoder', type=str, default='greedy', choices=['greedy', 'mbr', 'hybrid'], help='Eval-time decoder: greedy (argmax), token-wise MBR, or hybrid (TnT+Neural fusion)')
parser.add_argument('--mbr_weights', type=str, default='{"pos":2.0,"Subcategories":1.0,"Gender":1.0,"Number":1.0,"Case":1.0,"Article":0.5,"Proper Noun":1.0,"Degree":0.5,"Declension":0.5,"Mood":1.0,"Voice":0.5,"Tense":1.0,"Person":1.0,"Definiteness":0.5}', help='JSON dict of feature weights for MBR expected loss (only used with --eval_decoder mbr)')
parser.add_argument('--mbr_threshold', type=float, default=0.0, help='Optional abstention threshold for MBR: if max marginal < threshold, prefer no-feature value (only used with --eval_decoder mbr)')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for MBR logit calibration (only used with --eval_decoder mbr). Values > 1.0 soften probabilities.')
parser.add_argument('--hybrid_temperature', type=float, default=2.0, help='Temperature for neural logits in hybrid decoder (higher=more uniform, recommended 1.5-3.0)')
parser.add_argument('--hybrid_alpha', type=float, default=1.0, help='Weight for neural emissions in hybrid decoder')
parser.add_argument('--hybrid_beta', type=float, default=0.5, help='Weight for TnT emissions in hybrid decoder (0.0=neural only, 1.0=equal weight)')
parser.add_argument('--hybrid_lambda', type=float, default=0.7, help='Weight for TnT transitions in hybrid decoder')
parser.add_argument('--hybrid_entropy_gate', type=float, default=None, help='Entropy threshold for gating TnT fusion (None=always fuse)')
parser.add_argument('--tnt_model_path', type=str, default=None, help='Path to TnT model for hybrid decoder (auto-detected if not specified)')
parser.add_argument('--unconstrained_loss', type=str, default=None, choices=['unnormalized', 'normalized'],
    help='Ablation: compute loss on ALL feature groups. "unnormalized" sums all terms; "normalized" divides by number of active terms to match constrained loss magnitude.')
parser.add_argument('--full_train', action='store_true',
    help='Train on all data (no train/val split). Use with --fixed_epochs.')
parser.add_argument('--include_ood', action='store_true',
    help='Include OOD data in training (use with --full_train for release model).')
parser.add_argument('--fixed_epochs', type=int, default=None,
    help='Train for exactly N epochs (no early stopping). Overrides patience-based stopping.')
parser.add_argument('--save_huggingface', type=str, default=None,
    help='Save model in HuggingFace format to this directory after training.')
args = parser.parse_args()

# Parameters and Constants
TAGS_FILEPATH = 'data/Sosialurin-BRAGD_tags.csv'
CORPUS_FILEPATH = 'data/Sosialurin-BRAGD.tsv'
TOKENIZER_MODEL = 'vesteinn/ScandiBERT'
MAX_LEN = 128
BATCH_SIZE = args.batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.fixed_epochs if args.fixed_epochs else 100
FOLD = args.fold
MODE = args.mode
MODEL_TYPE = args.model_type
OPTIMIZER = args.optimizer
WARMUP_RATIO = args.warmup_ratio
WARMUP_STEPS = args.warmup_steps
EARLY_STOPPING_PATIENCE = 3 if not args.fixed_epochs else NUM_EPOCHS + 1  # effectively disable
EARLY_STOPPING_DELTA = 0.0001

def process_tag_features(tag_to_features, intervals):
    # Convert tag features to unique arrays
    list_of_tags = list(tag_to_features.values())
    tuples = [tuple(arr) for arr in list_of_tags]
    unique_tuples = set(tuples)
    unique_arrays = [np.array(tpl) for tpl in unique_tuples]
    print(f'Total number of unique tags {len(unique_arrays)}')

    # Generate word type masks
    word_type_masks = {}
    for word_type_number in range(15):
        word_labels = [array for array in unique_arrays if array[word_type_number] == 1]
        word_type_masks[word_type_number] = word_labels

    # Calculate allowed intervals for each word type
    dict_intervals = {}
    for label_word in range(15):
        labels_word = word_type_masks[label_word]
        arr_labels = np.array(labels_word)
        sum_labels = np.sum(arr_labels, axis=0)

        allowed_intervals = [interval for interval in intervals if np.sum(sum_labels[interval[0]:interval[1] + 1]) != 0]
        dict_intervals[label_word] = allowed_intervals

    return word_type_masks, dict_intervals

def initialize_model(tag_to_features, mode='multilabel', num_composite_tags=None):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'singlelabel':
        # Single-label classification: one class per unique composite tag
        model = XLMRobertaForTokenClassification.from_pretrained(
            TOKENIZER_MODEL,
            num_labels=num_composite_tags,
            problem_type="single_label_classification"
        )
    else:
        # Multi-label classification: N binary features
        model = XLMRobertaForTokenClassification.from_pretrained(
            TOKENIZER_MODEL,
            num_labels=len(tag_to_features[next(iter(tag_to_features))]),
            problem_type="multi_label_classification"
        )

    model.to(device)
    return tokenizer, device, model

def create_data_loaders(train_df, val_df, tokenizer, mode='multilabel', tag_to_id=None, features_to_tag=None):
    train_dataset = CustomDataset(
        train_df['sentences'].tolist(),
        train_df['tags'].tolist(),
        tokenizer,
        MAX_LEN,
        mode=mode,
        tag_to_id=tag_to_id,
        features_to_tag=features_to_tag
    )
    valid_dataset = CustomDataset(
        val_df['sentences'].tolist(),
        val_df['tags'].tolist(),
        tokenizer,
        MAX_LEN,
        mode=mode,
        tag_to_id=tag_to_id,
        features_to_tag=features_to_tag
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader

def log_loss(epoch, train_loss, valid_loss, accuracy_WT=False, f1_WT=False, accuracy_tot=False):
    if accuracy_WT and f1_WT:
        print(f'Epoch [{epoch + 1}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, '
              f'Accuracy WT: {accuracy_WT:.4f}, F1 WT: {f1_WT:.4f}, Accuracy tot: {accuracy_tot:.4f}')
    else:
        print(f'Epoch [{epoch + 1}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

def get_filtered_prediction(pred, labels, dict_intervals, device, use_true_labels=True):
    """
    Compute hierarchical loss for multilabel POS tagging.

    Args:
        pred: Model predictions (logits) for all features
        labels: True labels for all features
        dict_intervals: Dictionary mapping word types to their relevant feature intervals
        device: torch device
        use_true_labels: If True, use true word type to determine which features to compute loss on (TRAINING)
                        If False, use predicted word type (EVALUATION - matches inference behavior)
    """
    word_type_interval = np.arange(0, 15, 1)
    indexes = torch.from_numpy(word_type_interval).to(device)
    target_logits = torch.index_select(pred, 0, indexes)
    target_labels = torch.index_select(labels, 0, indexes)
    target_idx = target_labels.argmax(dim=0)
    loss_wordtype = F.cross_entropy(target_logits, target_idx)

    if use_true_labels:
        # TRAINING: Use TRUE word type to determine which features to train
        # This ensures we get gradients for features that SHOULD be present
        word_type_for_loss = target_idx.item()
    else:
        # EVALUATION: Use PREDICTED word type to match inference behavior
        # This measures how well the model does when it has to decide on its own
        softmax = torch.nn.Softmax(dim=0)
        probabilities = softmax(target_logits)
        word_type_for_loss = torch.argmax(probabilities).cpu().item()

    intervals = dict_intervals[word_type_for_loss]
    
    total_loss = 0
    for interval in intervals:
        indexes = torch.arange(interval[0], interval[-1] + 1, 1, device=device)
        target_logits = torch.index_select(pred, 0, indexes)
        target_labels = torch.index_select(labels, 0, indexes)
        target_idx = target_labels.argmax(dim=0)
        loss = F.cross_entropy(target_logits, target_idx)
        total_loss = total_loss + loss

    return total_loss + loss_wordtype

def get_unconstrained_prediction(pred, labels, intervals, device, normalize=False):
    """
    Compute unconstrained loss for multilabel POS tagging (ablation study).
    Iterates over ALL feature intervals regardless of word type.
    Unlike constrained loss, this does NOT skip irrelevant intervals --
    for all-zero gold intervals, argmax=0 provides a spurious target,
    testing whether the constrained masking helps learning.

    Args:
        normalize: If True, divide total interval loss by number of intervals
                   to match the per-token loss magnitude of constrained loss.
    """
    word_type_interval = np.arange(0, 15, 1)
    indexes = torch.from_numpy(word_type_interval).to(device)
    target_logits = torch.index_select(pred, 0, indexes)
    target_labels = torch.index_select(labels, 0, indexes)
    target_idx = target_labels.argmax(dim=0)
    loss_wordtype = F.cross_entropy(target_logits, target_idx)

    total_loss = 0
    n_intervals = 0
    for interval in intervals:
        indexes = torch.arange(interval[0], interval[-1] + 1, 1, device=device)
        target_labels_interval = torch.index_select(labels, 0, indexes)
        target_logits_interval = torch.index_select(pred, 0, indexes)
        # argmax on all-zeros returns 0 -- this is the intended spurious signal
        target_idx = target_labels_interval.argmax(dim=0)
        loss = F.cross_entropy(target_logits_interval, target_idx)
        total_loss = total_loss + loss
        n_intervals += 1

    if normalize and n_intervals > 0:
        # Count how many intervals the constrained version would use
        # (those with non-zero gold labels) and scale to match
        n_relevant = sum(
            1 for interval in intervals
            if labels[interval[0]:interval[-1] + 1].sum() > 0
        )
        if n_relevant > 0:
            total_loss = total_loss * (n_relevant / n_intervals)

    return total_loss + loss_wordtype

class CustomDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, max_len, mode='multilabel', tag_to_id=None, features_to_tag=None):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.tag_to_id = tag_to_id
        self.features_to_tag = features_to_tag

    def __len__(self):
        return len(self.sentences)

    def align_labels_with_tokens(self, tags, encoding):
        labels = []
        begin_token = []
        word_type = []
        word_ids = encoding.word_ids()
        last_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label = [0] * len(tags[0])
                begin_token.append(0)
                word_type.append(-9)
            elif word_id != last_word_id:
                label = tags[word_id]
                begin_token.append(1)
                word_type.append(np.where(label == 1)[0][0])
            else:
                label = labels[-1]
                begin_token.append(0)
                word_type.append(np.where(label == 1)[0][0])

            labels.append(label)
            last_word_id = word_id
        return labels, begin_token, word_type

    def __getitem__(self, index):
        sentence = self.sentences[index]
        tags = self.tags[index]

        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        aligned_labels, begin_token, word_types = self.align_labels_with_tokens(tags, encoding)

        if self.mode == 'singlelabel':
            # Convert feature vectors to composite tag IDs
            composite_ids = []
            for label in aligned_labels:
                if sum(label) == 0:  # Padding token
                    composite_ids.append(-100)  # Ignore index for loss calculation
                else:
                    composite_tag = self.features_to_tag.get(tuple(label), None)
                    if composite_tag is not None:
                        composite_ids.append(self.tag_to_id[composite_tag])
                    else:
                        composite_ids.append(-100)  # Unknown tag

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(composite_ids, dtype=torch.long),
                'begin_token': torch.tensor(begin_token, dtype=torch.long),
                'word_type': torch.tensor(word_types, dtype=torch.long),
                'original_labels': torch.from_numpy(np.array(aligned_labels)).long(),  # Convert to numpy first
            }
        else:
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.from_numpy(np.array(aligned_labels)).long(),  # Convert to numpy first
                'begin_token': torch.tensor(begin_token, dtype=torch.long),
                'word_type': torch.tensor(word_types, dtype=torch.long),
            }

def predict_classes(logits, attention_mask, begin_token, dict_intervals):
    softmax = torch.nn.Softmax(dim=0)
    prediction_sentence = []
    device = logits.device

    for i, mask_value in enumerate(attention_mask):
        # Skip CLS (i==0) and SEP tokens, only process begin tokens
        if mask_value == 1 and begin_token[i] == 1 and i > 0:
            pred_logits = logits[i]
            num_features = pred_logits.shape[0]
            prediction = torch.zeros(num_features, device=device)

            word_type_interval = torch.arange(0, 15, 1, device=device)
            target_logits = torch.index_select(pred_logits, 0, word_type_interval)
            probabilities = softmax(target_logits)
            predicted_class = torch.argmax(probabilities).item()
            prediction[predicted_class] = 1

            intervals = dict_intervals[predicted_class]
            for interval in intervals:
                indexes = torch.arange(interval[0], interval[-1] + 1, 1, device=device)
                target_logits = torch.index_select(pred_logits, 0, indexes)
                probabilities = softmax(target_logits)
                predicted_subclass = torch.argmax(probabilities).item()
                prediction[interval[0] + predicted_subclass] = 1

            prediction_sentence.append(prediction)

    return prediction_sentence

def _safe_softmax(x, dim=0):
    return torch.nn.functional.softmax(x, dim=dim)

def compute_marginals_for_word(pred_logits_i, dict_intervals):
    """
    Compute marginal probabilities for word-type (POS) and feature groups.

    Args:
        pred_logits_i: [NUM_FEATURES] logits for a single word (first subword position)
        dict_intervals: Dict mapping POS to allowed feature intervals

    Returns:
        Dict with 'pos' (probs, indices) and 'all_group_logits' (raw logits)
    """
    device = pred_logits_i.device
    # POS (first 15 indices)
    pos_idx = torch.arange(0, 15, device=device)
    pos_logits = torch.index_select(pred_logits_i, 0, pos_idx)
    pos_probs = _safe_softmax(pos_logits, dim=0)  # [15]

    return {
        'pos': (pos_probs, pos_idx),
        'all_group_logits': pred_logits_i  # will slice lazily per interval
    }

def decode_token_mbr(marginals, pred_logits_i, dict_intervals, mbr_weights, interval_to_name, mbr_threshold=0.0, debug=False):
    """
    Token-wise MBR: choose POS and subfeatures minimizing expected weighted 0-1 loss.

    Args:
        marginals: Dict from compute_marginals_for_word
        pred_logits_i: [NUM_FEATURES] logits for this word
        dict_intervals: Dict mapping POS to allowed intervals
        mbr_weights: Dict of feature weights for risk calculation
        interval_to_name: Dict mapping interval tuples to feature names
        mbr_threshold: Abstention threshold
        debug: Print debug information

    Returns:
        NUM_FEATURES-dim 0/1 prediction vector
    """
    device = pred_logits_i.device
    pred_vec = torch.zeros(pred_logits_i.shape[0], device=device)

    # 1) POS: weighted risk minimization
    # Expected loss for choosing POS j: w_pos * sum_i p(i) * 1(i != j) = w_pos * (1 - p(j))
    pos_probs, pos_idx = marginals['pos']
    pos_weight = mbr_weights.get('pos', 1.0)
    pos_risks = pos_weight * (1.0 - pos_probs)  # [15]
    pos_choice = torch.argmin(pos_risks).item()
    pred_vec[pos_choice] = 1

    if debug:
        print(f"      POS max prob: {pos_probs[pos_choice].item():.3f}")

    # 2) Subfeature groups allowed by this POS
    intervals = dict_intervals[pos_choice]
    abstained_count = 0

    for (start, end) in intervals:
        idxs = torch.arange(start, end + 1, device=device)
        group_logits = torch.index_select(pred_logits_i, 0, idxs)
        group_probs = _safe_softmax(group_logits, dim=0)

        # Get weight for this feature group
        feature_name = interval_to_name.get((start, end), 'Unknown')
        feature_weight = mbr_weights.get(feature_name, 1.0)

        # Weighted risk for each value: w_k * (1 - p(v))
        group_risks = feature_weight * (1.0 - group_probs)  # [group_size]
        min_risk, min_j = torch.min(group_risks, dim=0)
        chosen_offset = min_j.item()

        max_p = group_probs[chosen_offset].item()

        # Abstention: if confidence is low, prefer last index (typically "no-feature")
        if mbr_threshold > 0.0 and max_p < mbr_threshold:
            no_feat_idx = end  # convention: last index often represents "no-feature"
            pred_vec[no_feat_idx] = 1
            abstained_count += 1
            if debug:
                print(f"      {feature_name}: ABSTAINED (max_p={max_p:.3f} < {mbr_threshold})")
        else:
            pred_vec[start + chosen_offset] = 1
            if debug and max_p < 0.5:
                print(f"      {feature_name}: chose with prob {max_p:.3f}")

    if debug and abstained_count > 0:
        print(f"      Total abstentions: {abstained_count}/{len(intervals)}")

    return pred_vec

def mbr_decode_sentence(logits_seq, attention_mask_i, begin_token_i, dict_intervals, mbr_weights, interval_to_name, mbr_threshold):
    """
    Apply token-wise MBR to one sentence.

    Args:
        logits_seq: [seq_len, NUM_FEATURES] logits for full sequence (already temperature-scaled if applicable)
        attention_mask_i: [seq_len] attention mask
        begin_token_i: [seq_len] begin token indicators
        dict_intervals: Dict mapping POS to intervals
        mbr_weights: Feature weights
        interval_to_name: Dict mapping interval tuples to feature names
        mbr_threshold: Abstention threshold

    Returns:
        List of NUM_FEATURES-dim predictions for begin tokens only
    """
    out = []
    for tok_idx, mask_value in enumerate(attention_mask_i):
        if mask_value == 1 and begin_token_i[tok_idx] == 1 and tok_idx > 0:
            pred_logits_i = logits_seq[tok_idx]
            marginals = compute_marginals_for_word(pred_logits_i, dict_intervals)
            pred_vec = decode_token_mbr(marginals, pred_logits_i, dict_intervals, mbr_weights, interval_to_name, mbr_threshold)
            out.append(pred_vec)
    return out

def extract_tnt_statistics(tnt_model, tag_to_features, num_pos=15):
    """Extract transition matrix and unigram prior from TnT model.

    Args:
        tnt_model: Trained TnT model
        tag_to_features: Dict mapping composite tags to feature vectors
        num_pos: Number of POS tags (first 15 in feature vector)

    Returns:
        log_A: [num_pos, num_pos] transition log probabilities P(pos_t | pos_{t-1})
        log_pi: [num_pos] unigram log probabilities P(pos)
    """
    import numpy as np
    from collections import defaultdict

    # Aggregate counts by POS index
    pos_uni_counts = np.zeros(num_pos)
    pos_bi_counts = np.zeros((num_pos, num_pos))

    # Extract unigram counts
    for (tag, unk_flag), count in tnt_model._uni.items():
        if tag in tag_to_features:
            feature_vec = tag_to_features[tag]
            pos_idx = np.where(feature_vec[:num_pos] == 1)[0]
            if len(pos_idx) > 0:
                pos_uni_counts[pos_idx[0]] += count

    # Extract bigram counts
    for (prev_tag, prev_unk), freq_dist in tnt_model._bi.items():
        if prev_tag == 'BOS':
            # Start transitions - distribute to all tags
            continue
        if prev_tag in tag_to_features:
            prev_feature_vec = tag_to_features[prev_tag]
            prev_pos_idx = np.where(prev_feature_vec[:num_pos] == 1)[0]
            if len(prev_pos_idx) == 0:
                continue
            prev_pos = prev_pos_idx[0]

            for (curr_tag, curr_unk), count in freq_dist.items():
                if curr_tag in tag_to_features:
                    curr_feature_vec = tag_to_features[curr_tag]
                    curr_pos_idx = np.where(curr_feature_vec[:num_pos] == 1)[0]
                    if len(curr_pos_idx) > 0:
                        curr_pos = curr_pos_idx[0]
                        pos_bi_counts[prev_pos, curr_pos] += count

    # Normalize to get probabilities
    # Unigram: P(pos) = count(pos) / total_count
    total_uni = pos_uni_counts.sum()
    log_pi = np.log(np.maximum(pos_uni_counts / total_uni, 1e-10))

    # Bigram: P(curr | prev) = count(prev, curr) / count(prev)
    log_A = np.zeros((num_pos, num_pos))
    for prev_pos in range(num_pos):
        total_prev = pos_bi_counts[prev_pos, :].sum()
        if total_prev > 0:
            log_A[prev_pos, :] = np.log(np.maximum(pos_bi_counts[prev_pos, :] / total_prev, 1e-10))
        else:
            # Uniform fallback for unseen previous POS
            log_A[prev_pos, :] = np.log(1.0 / num_pos)

    return torch.from_numpy(log_A).float(), torch.from_numpy(log_pi).float()

def get_tnt_pos_posteriors(tnt_model, tokens, tag_to_features, device):
    """Get TnT posterior probabilities over POS tags for a sequence.

    Args:
        tnt_model: Trained TnT model
        tokens: List of token strings
        tag_to_features: Dict mapping tags to feature vectors
        device: PyTorch device

    Returns:
        log_posteriors: [seq_len, 15] log probabilities for each POS
    """
    import numpy as np
    from collections import defaultdict

    log_posteriors = torch.full((len(tokens), 15), -10.0, device=device)  # Small uniform prior

    for i, word in enumerate(tokens):
        # Aggregate emission probabilities by POS
        pos_counts = np.zeros(15)

        # Get word-tag distribution from TnT
        if word in tnt_model._wd:
            tag_dist = tnt_model._wd[word]
            total_count = sum(tag_dist.values())

            for composite_tag, count in tag_dist.items():
                if composite_tag in tag_to_features:
                    feature_vec = tag_to_features[composite_tag]
                    pos_idx_arr = np.where(feature_vec[:15] == 1)[0]
                    if len(pos_idx_arr) > 0:
                        pos_idx = pos_idx_arr[0]
                        pos_counts[pos_idx] += count

            # Convert to log probabilities
            if pos_counts.sum() > 0:
                probs = pos_counts / pos_counts.sum()
                log_probs = np.log(np.maximum(probs, 1e-10))
                log_posteriors[i] = torch.from_numpy(log_probs).float().to(device)
        else:
            # Unknown word: use TnT's greedy tag with high but not infinite confidence
            tagged = tnt_model.tag([word])
            if len(tagged) > 0:
                _, composite_tag = tagged[0]
                if composite_tag in tag_to_features:
                    feature_vec = tag_to_features[composite_tag]
                    pos_idx_arr = np.where(feature_vec[:15] == 1)[0]
                    if len(pos_idx_arr) > 0:
                        pos_idx = pos_idx_arr[0]
                        # Give unknown words lower confidence (log prob = -1 ≈ 37% confidence)
                        log_posteriors[i, pos_idx] = -1.0

    return log_posteriors

def hybrid_pos_viterbi_decode(neural_logits_pos, tnt_log_posteriors, log_A, log_pi,
                               temperature=2.0, alpha=1.0, beta=0.5, lambda_trans=0.7, entropy_gate=None):
    """Hybrid POS decoder using Viterbi with TnT+Neural fusion.

    Args:
        neural_logits_pos: [seq_len, 15] neural POS logits
        tnt_log_posteriors: [seq_len, 15] TnT POS log posteriors P(y|x)
        log_A: [15, 15] transition log probabilities P(y_t | y_{t-1})
        log_pi: [15] unigram log probabilities P(y)
        temperature: Temperature scaling for neural logits (higher = more uniform)
        alpha: Weight for neural emissions
        beta: Weight for TnT emissions
        lambda_trans: Weight for transitions
        entropy_gate: Entropy threshold for gating (None = always fuse)

    Returns:
        pos_sequence: [seq_len] predicted POS indices
    """
    device = neural_logits_pos.device
    T, P = neural_logits_pos.shape

    # 1) Calibrate neural logits with temperature
    neural_scaled = neural_logits_pos / temperature

    # 2) Convert TnT posteriors to pseudo-emissions: log p(y|x) - log p(y)
    log_pi_expanded = log_pi.unsqueeze(0).to(device)  # [1, P]
    tnt_pseudo_emit = tnt_log_posteriors - log_pi_expanded  # [T, P]

    # 3) Compute emissions with optional entropy gating
    if entropy_gate is not None:
        probs = torch.softmax(neural_scaled, dim=-1)
        # Entropy: H = -sum(p * log(p))
        H = -(probs * torch.clamp(probs, min=1e-9).log()).sum(dim=-1)  # [T]
        gate = (H > entropy_gate).float().unsqueeze(-1)  # [T, 1]
        emit = alpha * neural_scaled + gate * (beta * tnt_pseudo_emit)
    else:
        emit = alpha * neural_scaled + beta * tnt_pseudo_emit

    # Viterbi forward pass
    dp = torch.full((T, P), -1e9, device=device)
    backpointers = torch.zeros((T, P), dtype=torch.long, device=device)

    # Initialize
    dp[0] = emit[0]

    # Forward
    for t in range(1, T):
        # Broadcast: dp[t-1] is [P], log_A is [P,P]
        # We want: dp[t-1, prev_pos] + lambda_trans * log_A[prev_pos, curr_pos]
        scores = dp[t-1].unsqueeze(1) + lambda_trans * log_A  # [P, P]
        best_prev_scores, best_prev_pos = torch.max(scores, dim=0)  # [P]
        dp[t] = best_prev_scores + emit[t]
        backpointers[t] = best_prev_pos

    # Backtrack
    pos_sequence = []
    last_pos = torch.argmax(dp[-1]).item()
    pos_sequence.append(last_pos)

    for t in range(T-1, 0, -1):
        last_pos = backpointers[t, last_pos].item()
        pos_sequence.append(last_pos)

    pos_sequence.reverse()
    return pos_sequence

def hybrid_decode_sentence(neural_logits, attention_mask, begin_tokens, tokens, tnt_model, tag_to_features,
                           dict_intervals, log_A, log_pi, temperature, alpha, beta, lambda_trans, entropy_gate, device):
    """Decode a sentence using hybrid TnT+Neural approach.

    Args:
        neural_logits: [seq_len, NUM_FEATURES] neural model logits
        attention_mask: [seq_len] attention mask
        begin_tokens: [seq_len] begin token indicators
        tokens: List of token strings (for TnT)
        tnt_model: Trained TnT model
        tag_to_features: Dict mapping tags to features
        dict_intervals: Dict mapping POS to allowed intervals
        log_A: [15, 15] transition matrix
        log_pi: [15] unigram prior
        temperature: Temperature for neural logits
        alpha: Neural emission weight
        beta: TnT emission weight
        lambda_trans: Transition weight
        entropy_gate: Entropy gating threshold
        device: PyTorch device

    Returns:
        List of NUM_FEATURES-dim predictions for begin tokens only
    """
    # Extract begin token positions and neural logits
    begin_indices = []
    begin_neural_logits = []

    for i, (mask, is_begin) in enumerate(zip(attention_mask, begin_tokens)):
        if mask == 1 and is_begin == 1 and i > 0:  # Skip CLS
            begin_indices.append(i)
            begin_neural_logits.append(neural_logits[i])

    if len(begin_indices) == 0:
        return []

    # Map to original tokens: each begin token corresponds to one original token in order
    num_begin = len(begin_indices)
    begin_toks = tokens[:num_begin]  # Take first N original tokens

    # Stack neural logits for begin tokens
    stacked_logits = torch.stack(begin_neural_logits)  # [num_begin, NUM_FEATURES]
    neural_pos_logits = stacked_logits[:, :15]  # First 15 are POS

    # Get TnT posteriors
    tnt_log_posts = get_tnt_pos_posteriors(tnt_model, begin_toks, tag_to_features, device)

    # Decode POS with hybrid Viterbi
    pos_sequence = hybrid_pos_viterbi_decode(
        neural_pos_logits, tnt_log_posts, log_A, log_pi,
        temperature, alpha, beta, lambda_trans, entropy_gate
    )

    # Decode features using neural model, gated by hybrid POS
    num_features = stacked_logits.shape[1]
    predictions = []
    for i, pos_idx in enumerate(pos_sequence):
        pred_vec = torch.zeros(num_features, device=device)
        pred_vec[pos_idx] = 1  # Set POS from hybrid

        # Get allowed feature intervals for this POS
        intervals = dict_intervals[pos_idx]

        # Decode each feature group with greedy argmax
        full_logits = stacked_logits[i]
        for (start, end) in intervals:
            idxs = torch.arange(start, end + 1, device=device)
            group_logits = torch.index_select(full_logits, 0, idxs)
            group_probs = torch.softmax(group_logits, dim=0)
            chosen_offset = torch.argmax(group_probs).item()
            pred_vec[start + chosen_offset] = 1

        predictions.append(pred_vec)

    return predictions

def calculate_composite_tag_accuracy(pred_labels, true_labels, features_to_tag):
    """Calculate composite tag accuracy by comparing full feature vectors."""
    pred_composite = []
    true_composite = []

    for pred, true in zip(pred_labels, true_labels):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(true, torch.Tensor):
            true = true.cpu().numpy()

        pred_tag = vector_to_composite_tag(pred, features_to_tag)
        true_tag = vector_to_composite_tag(true, features_to_tag)

        if pred_tag is not None and true_tag is not None:
            pred_composite.append(pred_tag)
            true_composite.append(true_tag)

    if len(pred_composite) == 0:
        return [], [], 0.0

    accuracy = np.mean(np.array(pred_composite) == np.array(true_composite))
    return pred_composite, true_composite, accuracy

def calculate_accuracy_singlelabel(pred_tag_ids, gold_tag_ids, id_to_tag, tag_to_features, intervals):
    """
    Calculate per-label accuracy for singlelabel mode by parsing composite tags.

    Args:
        pred_tag_ids: List of predicted tag IDs
        gold_tag_ids: List of gold tag IDs
        id_to_tag: Dict mapping tag ID to tag string
        tag_to_features: Dict mapping tag string to feature vector
        intervals: List of (start, end) tuples for each morphological feature

    Returns:
        accuracy_single_interval: Dict mapping interval tuples to list of [pred_val, gold_val] pairs
    """
    accuracy_single_interval = defaultdict(list)

    for pred_id, gold_id in zip(pred_tag_ids, gold_tag_ids):
        # Convert IDs to tag strings
        pred_tag = id_to_tag.get(pred_id, None)
        gold_tag = id_to_tag.get(gold_id, None)

        if pred_tag is None or gold_tag is None:
            continue

        # Convert tags to feature vectors
        pred_features = tag_to_features.get(pred_tag, None)
        gold_features = tag_to_features.get(gold_tag, None)

        if pred_features is None or gold_features is None:
            continue

        # For each interval (morphological feature), extract the value
        for interval in intervals:
            start, end = interval[0], interval[-1] if isinstance(interval, (list, tuple)) else interval

            # Extract feature values from the interval range
            pred_interval_features = pred_features[start:end+1]
            gold_interval_features = gold_features[start:end+1]

            # Find which position has value 1 (one-hot encoded)
            pred_val = np.argmax(pred_interval_features) if np.sum(pred_interval_features) > 0 else -1
            gold_val = np.argmax(gold_interval_features) if np.sum(gold_interval_features) > 0 else -1

            # Track if gold has a valid value (evaluate based on what SHOULD be predicted)
            # If pred doesn't have this feature but gold does, treat as incorrect prediction
            if gold_val >= 0:
                # If predicted tag lacks this feature, use a special "wrong" value
                # that will never match gold_val (use a large negative number)
                actual_pred_val = pred_val if pred_val >= 0 else -999
                accuracy_single_interval[interval].append([actual_pred_val, gold_val])

    return accuracy_single_interval


def calculate_accuracy(predictions_batch, labels, dict_intervals, device, begin_tokens=None):
    refs_WT = []
    preds_WT = []
    accuracy_all_batch = []
    accuracy_single_WT = defaultdict(list)
    accuracy_single_interval = defaultdict(list)

    for t, label in enumerate(labels):
        pred_sentence = predictions_batch[t]

        # Extract labels only for begin tokens (first subword of each word)
        # This matches what predict_classes returns
        if begin_tokens is not None:
            begin_token_mask = begin_tokens[t]
            label_sentence = []
            for idx, (lbl, is_begin) in enumerate(zip(label, begin_token_mask)):
                # Skip CLS (first token) and SEP (last token), only keep begin tokens
                if is_begin == 1 and idx > 0:  # idx > 0 skips CLS
                    label_sentence.append(lbl)
            # Convert to tensor
            if len(label_sentence) > 0:
                label_sentence = torch.stack(label_sentence)
            else:
                continue
        else:
            # Fallback to old behavior (but this is incorrect for multi-subword words)
            label_sentence = label[1:len(pred_sentence)+1]

        sentence_refs = []
        sentence_preds = []

        for ind, pred in enumerate(pred_sentence):
            if len(pred) > 1:
                word_type_interval = np.arange(0, 15, 1)
                indexes = torch.from_numpy(word_type_interval).to(device)
                target_preds = torch.index_select(pred, 0, indexes)
                target_labels = torch.index_select(label_sentence[ind], 0, indexes)
                _, pred_index = torch.max(target_preds, 0)
                _, true_index = torch.max(target_labels, 0)
                preds_WT.append(pred_index.item())
                refs_WT.append(true_index.item())

                # Collect predictions and references per class
                accuracy_single_WT[true_index.item()].append((pred_index.item(), true_index.item()))

                # For sentence-level accuracy
                sentence_preds.append(pred_index.item())
                sentence_refs.append(true_index.item())

                intervals = dict_intervals[true_index.item()]
                for inter in intervals:
                    subclass_interval = np.arange(inter[0], inter[-1]+1, 1)
                    indexes = torch.from_numpy(subclass_interval).to(device)
                    target_preds = torch.index_select(pred, 0, indexes)
                    target_labels = torch.index_select(label_sentence[ind], 0, indexes)
                    _, pred_sub_index = torch.max(target_preds, 0)
                    _, true_sub_index = torch.max(target_labels, 0)
                    accuracy_single_interval[inter].append([int(pred_sub_index.item()), int(true_sub_index.item())])

        # Compute accuracy for the sentence
        if sentence_refs:
            sentence_accuracy = np.mean(np.array(sentence_preds) == np.array(sentence_refs))
            accuracy_all_batch.append(sentence_accuracy)

    return preds_WT, refs_WT, accuracy_all_batch, accuracy_single_WT, accuracy_single_interval


def evaluate_model(model, dataloader, device, MODE, tag_to_id, id_to_tag, tag_to_features,
                   features_to_tag, intervals, dict_intervals, name_intervals, all_word_classes,
                   eval_decoder='greedy', temperature=1.0, mbr_threshold=0.0, mbr_weights=None,
                   interval_to_name=None, description="Evaluation"):
    """
    Unified evaluation function for validation and OOD data.

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    valid_loss = 0
    overall_accuracy = []
    preds_WT = []
    refs_WT = []
    acc_single_WT = defaultdict(list)
    acc_single_intr = defaultdict(list)

    # For composite tag accuracy (both modes)
    all_pred_composite_tags = []
    all_true_composite_tags = []

    # For singlelabel per-interval metrics
    all_pred_tag_ids = []
    all_gold_tag_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {description}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            begin_tokens = batch["begin_token"].to(device)

            # MODE-SPECIFIC EVALUATION
            if MODE == 'singlelabel':
                # SINGLE-LABEL MODE
                outputs = model(input_ids, attention_mask, labels=labels)
                predictions = outputs.logits
                loss = outputs.loss.item()
                valid_loss += loss

                # Get predicted class IDs
                pred_class_ids = torch.argmax(predictions, dim=-1)  # [batch, seq_len]

                # Collect predictions and labels for begin tokens only
                for idx in range(input_ids.size(0)):
                    for widx in range(input_ids.size(1)):
                        if begin_tokens[idx][widx] == 1:
                            pred_id = pred_class_ids[idx][widx].item()
                            true_label = batch["original_labels"][idx][widx].cpu().numpy()

                            # Convert prediction ID to composite tag
                            if pred_id < len(id_to_tag):
                                pred_tag = id_to_tag[pred_id]
                                true_tag = vector_to_composite_tag(true_label, features_to_tag)

                                if pred_tag and true_tag:
                                    all_pred_composite_tags.append(pred_tag)
                                    all_true_composite_tags.append(true_tag)

                                    # Extract word type from feature vectors
                                    true_word_type = np.where(true_label == 1)[0][0]
                                    pred_features = tag_to_features[pred_tag]
                                    pred_word_type = np.where(pred_features == 1)[0][0]

                                    preds_WT.append(pred_word_type)
                                    refs_WT.append(true_word_type)

                                    # Populate acc_single_WT for per-class accuracy calculation
                                    acc_single_WT[true_word_type].append((pred_word_type, true_word_type))

                                    # Collect tag IDs for per-interval metrics
                                    true_tag_id = tag_to_id.get(true_tag, -1)
                                    if true_tag_id >= 0:
                                        all_pred_tag_ids.append(pred_id)
                                        all_gold_tag_ids.append(true_tag_id)

            else:
                # MULTI-LABEL MODE
                outputs = model(input_ids, attention_mask)
                predictions = outputs.logits
                loss = 0
                total_valid_tokens = 0

                predictions_batch = []
                for idx, sentence in enumerate(input_ids):
                    # Use MBR or greedy decoding based on flag
                    if eval_decoder == 'mbr':
                        # Apply temperature scaling for MBR
                        temp_scaled_logits = predictions[idx] / temperature
                        predicted_labels = mbr_decode_sentence(
                            temp_scaled_logits, attention_mask[idx], begin_tokens[idx],
                            dict_intervals, mbr_weights, interval_to_name, mbr_threshold
                        )
                    else:
                        predicted_labels = predict_classes(predictions[idx], attention_mask[idx], begin_tokens[idx], dict_intervals)
                    predictions_batch.append(predicted_labels)

                # Calculate loss for each token (EVALUATION - use predicted word type)
                for idx, sentence in enumerate(input_ids):
                    for widx, word in enumerate(sentence):
                        if begin_tokens[idx][widx] == 0:
                            continue
                        true_word_type = batch["word_type"][idx][widx]
                        if true_word_type == -9:
                            continue
                        if args.unconstrained_loss:
                            loss += get_unconstrained_prediction(predictions[idx][widx], labels[idx][widx], intervals, device, normalize=(args.unconstrained_loss == 'normalized'))
                        else:
                            loss += get_filtered_prediction(predictions[idx][widx], labels[idx][widx], dict_intervals, device, use_true_labels=False)
                        total_valid_tokens += 1

                preds_WT_batch, refs_WT_batch, accuracy_all_classes_batch, accuracy_single_WT_batch, accuracy_single_interval = calculate_accuracy(predictions_batch, labels, dict_intervals, device, begin_tokens)

                preds_WT.extend(preds_WT_batch)
                refs_WT.extend(refs_WT_batch)
                overall_accuracy.extend(accuracy_all_classes_batch)

                for key, value in accuracy_single_WT_batch.items():
                    acc_single_WT[key].extend(value)

                for key, value in accuracy_single_interval.items():
                    acc_single_intr[key].extend(value)

                # Calculate composite tag accuracy for multilabel
                for idx, pred_batch in enumerate(predictions_batch):
                    # Get true labels for begin tokens (matching predict_classes behavior)
                    true_labels_for_sentence = []
                    for widx in range(labels.size(1)):
                        if begin_tokens[idx][widx] == 1:
                            true_label = labels[idx][widx].cpu().numpy()
                            true_labels_for_sentence.append(true_label)

                    # Match predictions with true labels
                    for pred, true in zip(pred_batch, true_labels_for_sentence):
                        if len(pred) > 1:  # Only count valid predictions
                            pred_tag = vector_to_composite_tag(pred, features_to_tag)
                            true_tag = vector_to_composite_tag(true, features_to_tag)

                            if pred_tag and true_tag:
                                all_pred_composite_tags.append(pred_tag)
                                all_true_composite_tags.append(true_tag)

                loss = loss / total_valid_tokens if total_valid_tokens > 0 else loss
                valid_loss += loss.item()

    avg_loss_valid = valid_loss / len(dataloader)

    # Compute per-interval metrics for singlelabel mode
    if MODE == 'singlelabel' and len(all_pred_tag_ids) > 0 and len(all_gold_tag_ids) > 0:
        interval_metrics = calculate_accuracy_singlelabel(
            all_pred_tag_ids, all_gold_tag_ids, id_to_tag, tag_to_features, intervals
        )
        # Merge into acc_single_intr
        for interval, values in interval_metrics.items():
            acc_single_intr[interval].extend(values)

    # Compute overall accuracy for word types
    accuracy_metric = evaluate.load("accuracy")
    accuracy_WT = accuracy_metric.compute(references=refs_WT, predictions=preds_WT)
    accuracy_tot = np.mean(overall_accuracy) if overall_accuracy else 0.0

    # Compute F1 scores
    f1_metric = evaluate.load('f1')

    # Exclude outlier classes with very few examples from macro F1
    excluded_classes = [9, 11]  # Unanalyzed word, Web email or address
    filtered_refs_WT = []
    filtered_preds_WT = []
    for ref, pred in zip(refs_WT, preds_WT):
        if ref not in excluded_classes:
            filtered_refs_WT.append(ref)
            filtered_preds_WT.append(pred)

    f1_WT = f1_metric.compute(references=filtered_refs_WT, predictions=filtered_preds_WT, average='macro')['f1'] if len(filtered_refs_WT) > 0 else 0.0

    # COMPOSITE TAG METRICS (for both modes)
    composite_accuracy = 0.0
    composite_micro_f1 = 0.0
    composite_macro_f1 = 0.0
    if len(all_pred_composite_tags) > 0 and len(all_true_composite_tags) > 0:
        composite_accuracy = np.mean(np.array(all_pred_composite_tags) == np.array(all_true_composite_tags))

        # Convert composite tags to IDs for F1 computation
        pred_tag_ids = [tag_to_id[tag] for tag in all_pred_composite_tags if tag in tag_to_id]
        true_tag_ids = [tag_to_id[tag] for tag in all_true_composite_tags if tag in tag_to_id]

        if len(pred_tag_ids) > 0 and len(true_tag_ids) > 0:
            composite_micro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='micro')['f1']
            composite_macro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='macro')['f1']

    # Compute per-class F1 scores for word types
    unique_labels = sorted(set(refs_WT + preds_WT))
    f1_result = f1_metric.compute(
        references=refs_WT,
        predictions=preds_WT,
        average=None,
        labels=unique_labels,
    )
    class_f1_scores = dict(zip(unique_labels, f1_result['f1']))

    # Assign per-class F1 scores, setting 0.0 for classes not present
    acc_single_WT_f1 = {label: class_f1_scores.get(label, 0.0) for label in all_word_classes}

    # Per-interval F1 scores
    acc_single_intr_f1 = {}
    for key in acc_single_intr.keys():
        values = acc_single_intr[key]
        preds = [tup[0] for tup in values]
        refs = [tup[1] for tup in values]
        if len(set(refs)) > 1:
            f1_result = f1_metric.compute(references=refs, predictions=preds, average='macro')
            acc_single_intr_f1[key] = f1_result['f1']
        else:
            acc_single_intr_f1[key] = 0.0  # Avoid division by zero

    # Process accuracies for individual word types and intervals
    for key in acc_single_WT:
        correct_preds = sum(1 for pred, ref in acc_single_WT[key] if pred == ref)
        acc = correct_preds / len(acc_single_WT[key])
        acc_single_WT[key] = acc

    for key in acc_single_intr.keys():
        values = acc_single_intr[key]
        preds = [tup[0] for tup in values]
        refs = [tup[1] for tup in values]
        acc_result = accuracy_metric.compute(references=refs, predictions=preds)
        acc_single_intr[key] = acc_result['accuracy']

    # Return all metrics in a dictionary
    return {
        'loss': avg_loss_valid,
        'composite_accuracy': composite_accuracy,
        'composite_micro_f1': composite_micro_f1,
        'composite_macro_f1': composite_macro_f1,
        'word_class_accuracy': accuracy_WT['accuracy'],
        'word_class_macro_f1': f1_WT,
        'per_word_class_accuracy': dict(sorted(acc_single_WT.items())),
        'per_word_class_f1': acc_single_WT_f1,
        'per_interval_accuracy': {name_intervals[key]: acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
        'per_interval_f1': {name_intervals[key]: acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
        'num_sentences': len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else 0,
    }


def main():
    global MODE  # Allow modification of MODE global variable

    # Check for evaluation-only mode and auto-detect mode from checkpoint path
    if args.evaluate_only:
        if not args.checkpoint_path:
            raise ValueError("--evaluate_only requires --checkpoint_path")
        if not os.path.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {args.checkpoint_path}")

        # Auto-detect mode from checkpoint path if neural model
        if MODEL_TYPE == 'neural':
            checkpoint_path_lower = args.checkpoint_path.lower()
            if 'singlelabel' in checkpoint_path_lower:
                detected_mode = 'singlelabel'
            elif 'multilabel' in checkpoint_path_lower:
                detected_mode = 'multilabel'
            else:
                # Fallback: try to infer from checkpoint
                print("[WARNING] Could not auto-detect mode from path, trying to load checkpoint...")
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    # Check the output layer size
                    classifier_weight_key = 'classifier.weight'
                    if classifier_weight_key in checkpoint['model_state_dict']:
                        num_labels = checkpoint['model_state_dict'][classifier_weight_key].shape[0]
                        if num_labels < 100:
                            detected_mode = 'multilabel'
                        else:
                            detected_mode = 'singlelabel'
                        print(f"[INFO] Detected {num_labels} output labels -> mode: {detected_mode}")
                    else:
                        detected_mode = MODE  # Use provided mode as fallback
                        print(f"[WARNING] Could not detect mode from checkpoint, using provided mode: {MODE}")
                else:
                    detected_mode = MODE

            # Override MODE if it was auto-detected differently
            if detected_mode != MODE:
                print(f"[INFO] Auto-detected mode '{detected_mode}' from checkpoint (you specified '{MODE}')")
                MODE = detected_mode

        print("\n" + "="*80)
        print(f"POS Tagger Evaluation Only - Mode: {MODE.upper()} | Model: {MODEL_TYPE.upper()} | Fold: {FOLD}")
        print(f"Checkpoint: {args.checkpoint_path}")
        print("="*80 + "\n")

    # Validate arguments
    if MODEL_TYPE == 'tnt' and MODE != 'singlelabel':
        raise ValueError("TnT tagger only supports singlelabel mode. Use --mode singlelabel with --model_type tnt")

    # Check that only one of warmup_ratio or warmup_steps is set
    if WARMUP_RATIO > 0 and WARMUP_STEPS > 0:
        raise ValueError("Cannot specify both --warmup_ratio and --warmup_steps. Please use only one.")

    if not args.evaluate_only:
        print("\n" + "="*80)
        loss_type = f"UNCONSTRAINED-{args.unconstrained_loss.upper()}" if args.unconstrained_loss else "CONSTRAINED"
        print(f"POS Tagger Training - Mode: {MODE.upper()} ({loss_type}) | Model: {MODEL_TYPE.upper()} | Fold: {FOLD}")
        print(f"Train Batch Size: {BATCH_SIZE} | Eval Batch Size: {EVAL_BATCH_SIZE}")
        if MODEL_TYPE == 'neural':
            if OPTIMIZER in ['adam', 'adamw']:
                warmup_str = ""
                if WARMUP_STEPS > 0:
                    warmup_str = f" | Warmup: {WARMUP_STEPS} steps"
                elif WARMUP_RATIO > 0:
                    warmup_str = f" | Warmup: {WARMUP_RATIO*100:.1f}%"
                print(f"Optimizer: {OPTIMIZER.upper()} | LR: {LEARNING_RATE}{warmup_str}")
            else:
                print(f"Optimizer: {OPTIMIZER.upper()} | LR: adaptive")
        print("="*80 + "\n")

        # Only initialize wandb for neural models
        if MODEL_TYPE == 'neural':
            if HAS_WANDB:
                wandb.init(project='POS_tagger', name=f'Fold: {FOLD}')

    # Load and process corpus
    print("Loading corpus and processing tags...")
    sentences, sentence_tags, tag_to_features = load_and_process_corpus(TAGS_FILEPATH, CORPUS_FILEPATH)
    NUM_FEATURES = len(next(iter(tag_to_features.values())))
    print(f"[OK] Loaded {len(sentences)} sentences ({NUM_FEATURES} features per tag)")

    # Create tag mappings
    print("Creating tag mappings...")
    features_to_tag, tag_to_id, id_to_tag = create_tag_mappings(tag_to_features)
    num_composite_tags = len(tag_to_id)
    print(f'[OK] Created mappings for {num_composite_tags} unique composite tags')

    # Updated intervals for BRAGD tagset (73 features)
    intervals = (
        (15, 29),  # Subcategories (D,B,E,I,P,Q,N,G,R,X,S,C,O,T,s)
        (30, 33),  # Gender (M,F,N,g)
        (34, 36),  # Number (S,P,n)
        (37, 41),  # Case (N,A,D,G,c)
        (42, 43),  # Article/No-Article (Article,a)
        (44, 45),  # Proper/Not Proper Noun (Proper,r)
        (46, 50),  # Degree (P,C,S,A,d)
        (51, 53),  # Declension (S,W,e)
        (54, 60),  # Mood (I,M,N,S,P,E,U)
        (61, 63),  # Voice (A,M,v)
        (64, 66),  # Tense (P,A,t)
        (67, 70),  # Person (1,2,3,p)
        (71, 72),  # Definiteness (D,I)
        )

    name_intervals = {
        (15, 29): 'Subcategories',
        (30, 33): 'Gender',
        (34, 36): 'Number',
        (37, 41): 'Case',
        (42, 43): 'Article',
        (44, 45): 'Proper Noun',
        (46, 50): 'Degree',
        (51, 53): 'Declension',
        (54, 60): 'Mood',
        (61, 63): 'Voice',
        (64, 66): 'Tense',
        (67, 70): 'Person',
        (71, 72): 'Definiteness'
    }

    name_word_class = {
        0: 'Word Class Noun',
        1: 'Word Class Adjective',
        2: 'Word Class Pronoun',
        3: 'Word Class Number',
        4: 'Word Class Verbs',
        5: 'Word Class Participle',
        6: 'Word Class Adverb',
        7: 'Word Class Conjunctions',
        8: 'Word Class Foreign words',
        9: 'Word Class Unanalyzed word',
        10: 'Word Class Abbreviation',
        11: 'Word Class Web email or address',
        12: 'Word Class Punctuation',
        13: 'Word Class Symbol',
        14: 'Word Class R Article'
    }

    word_type_masks, dict_intervals = process_tag_features(tag_to_features, intervals)

    # Parse MBR weights for eval-time decoding
    mbr_weights = json.loads(args.mbr_weights)

    # Create reverse mapping: interval -> name (for MBR weight lookup)
    interval_to_name = {interval: name for interval, name in name_intervals.items()}

    # Split data
    if args.full_train:
        print(f"\nFull training mode: using ALL {len(sentences)} sentences...")
        all_indexes = list(range(len(sentences)))
        train_df, _ = train_test_split_data(sentences, sentence_tags, all_indexes, [])
        if args.include_ood and args.ood_data_path:
            print(f"Including OOD data from {args.ood_data_path}...")
            ood_sentences, ood_tags = load_ood_data(args.ood_data_path, tag_to_features)
            ood_df = pd.DataFrame({'sentences': ood_sentences, 'tags': ood_tags})
            train_df = pd.concat([train_df, ood_df], ignore_index=True)
            print(f"[OK] Combined training set: {len(train_df)} sentences (corpus + OOD)")
        else:
            print(f"[OK] Train: {len(train_df)} sentences (full corpus)")
        val_df = train_df  # Use train as val for monitoring (no early stopping)
    else:
        print(f"\nPreparing data splits for fold {FOLD}...")
        train_indexes, val_indexes = load_split_indices(FOLD)
        train_df, val_df = train_test_split_data(sentences, sentence_tags, train_indexes, val_indexes)
        print(f"[OK] Train: {len(train_df)} sentences | Val: {len(val_df)} sentences")

    # ============================================================================
    # TnT BASELINE MODEL
    # ============================================================================
    if MODEL_TYPE == 'tnt':
        from nltk.tag import tnt

        print(f"\nInitializing TnT tagger...")

        # Convert data to TnT format
        train_tnt = prepare_tnt_data(
            train_df['sentences'].tolist(),
            train_df['tags'].tolist(),
            features_to_tag
        )
        val_tnt = prepare_tnt_data(
            val_df['sentences'].tolist(),
            val_df['tags'].tolist(),
            features_to_tag
        )
        print(f"[OK] Data converted - Train: {len(train_tnt)} | Val: {len(val_tnt)} sentences")

        # Create checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f'fold_{FOLD}_tnt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, 'tnt_model.pkl')

        # Train or load TnT model
        if not args.evaluate_only:
            print("\nTraining TnT tagger...")
            tagger = tnt.TnT()
            tagger.train(train_tnt)
            print("[OK] Training complete")

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(tagger, f)
            print(f"[OK] Model saved to {model_path}")
        else:
            if not os.path.exists(model_path):
                raise ValueError(f"TnT model not found: {model_path}")
            print(f"Loading TnT model from {model_path}...")
            with open(model_path, 'rb') as f:
                tagger = pickle.load(f)
            print("[OK] TnT model loaded")

        # Evaluate on validation set
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}\n")

        all_pred_tags = []
        all_true_tags = []
        preds_WT = []
        refs_WT = []
        acc_single_WT = defaultdict(list)

        # For per-interval metrics
        all_pred_tag_ids = []
        all_gold_tag_ids = []

        for sent_tnt in tqdm(val_tnt, desc="Evaluating"):
            words = [word for word, tag in sent_tnt]
            true_tags = [tag for word, tag in sent_tnt]

            # Predict
            pred_sent = tagger.tag(words)
            pred_tags = [tag for word, tag in pred_sent]

            # Collect predictions
            for pred_tag, true_tag in zip(pred_tags, true_tags):
                all_pred_tags.append(pred_tag)
                all_true_tags.append(true_tag)

                # Extract word types
                if pred_tag in tag_to_features and true_tag in tag_to_features:
                    pred_features = tag_to_features[pred_tag]
                    true_features = tag_to_features[true_tag]

                    pred_word_type = np.where(pred_features == 1)[0][0]
                    true_word_type = np.where(true_features == 1)[0][0]

                    preds_WT.append(pred_word_type)
                    refs_WT.append(true_word_type)
                    acc_single_WT[true_word_type].append((pred_word_type, true_word_type))

                    # Collect tag IDs for per-interval metrics
                    if pred_tag in tag_to_id and true_tag in tag_to_id:
                        all_pred_tag_ids.append(tag_to_id[pred_tag])
                        all_gold_tag_ids.append(tag_to_id[true_tag])

        # Compute metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load('f1')

        # Composite tag metrics
        composite_accuracy = np.mean(np.array(all_pred_tags) == np.array(all_true_tags))

        # Only keep pairs where both pred and true tags are in tag_to_id
        pred_tag_ids = []
        true_tag_ids = []
        for pred_tag, true_tag in zip(all_pred_tags, all_true_tags):
            if pred_tag in tag_to_id and true_tag in tag_to_id:
                pred_tag_ids.append(tag_to_id[pred_tag])
                true_tag_ids.append(tag_to_id[true_tag])

        composite_micro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='micro')['f1']
        composite_macro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='macro')['f1']

        print(f"Composite Tag Metrics:")
        print(f"  Accuracy:  {composite_accuracy:.4f}")
        print(f"  Micro F1:  {composite_micro_f1:.4f}")
        print(f"  Macro F1:  {composite_macro_f1:.4f}")

        # Word class metrics
        if len(preds_WT) > 0:
            accuracy_WT = accuracy_metric.compute(references=refs_WT, predictions=preds_WT)['accuracy']

            # Exclude outlier classes with very few examples from macro F1
            excluded_classes = [9, 11]  # Unanalyzed word, Web email or address
            filtered_refs_WT = []
            filtered_preds_WT = []
            for ref, pred in zip(refs_WT, preds_WT):
                if ref not in excluded_classes:
                    filtered_refs_WT.append(ref)
                    filtered_preds_WT.append(pred)

            f1_WT = f1_metric.compute(references=filtered_refs_WT, predictions=filtered_preds_WT, average='macro')['f1'] if len(filtered_refs_WT) > 0 else 0.0

            print(f"\nWord Class Metrics:")
            print(f"  Accuracy:  {accuracy_WT:.4f}")
            print(f"  Macro F1:  {f1_WT:.4f}")

            # Per-class accuracies
            for key in acc_single_WT:
                correct_preds = sum(1 for pred, ref in acc_single_WT[key] if pred == ref)
                acc = correct_preds / len(acc_single_WT[key])
                acc_single_WT[key] = acc

            # Per-class F1
            unique_labels = sorted(set(refs_WT + preds_WT))
            f1_result = f1_metric.compute(references=refs_WT, predictions=preds_WT, average=None, labels=unique_labels)
            class_f1_scores = dict(zip(unique_labels, f1_result['f1']))
            acc_single_WT_f1 = {label: class_f1_scores.get(label, 0.0) for label in range(15)}

            print(f"\n  Per-Word-Class Accuracy:")
            for key in sorted(acc_single_WT.keys()):
                if key in name_word_class:
                    print(f"    {name_word_class[key]:30s}: {acc_single_WT[key]:.4f}")

            print(f"\n  Per-Word-Class F1:")
            for key in sorted(acc_single_WT_f1.keys()):
                if acc_single_WT_f1[key] > 0 and key in name_word_class:
                    print(f"    {name_word_class[key]:30s}: {acc_single_WT_f1[key]:.4f}")

        # Compute per-interval metrics
        acc_single_intr = {}
        acc_single_intr_f1 = {}
        if len(all_pred_tag_ids) > 0 and len(all_gold_tag_ids) > 0:
            interval_metrics = calculate_accuracy_singlelabel(
                all_pred_tag_ids, all_gold_tag_ids, id_to_tag, tag_to_features, intervals
            )

            # Calculate accuracy and F1 for each interval
            for interval, values in interval_metrics.items():
                preds = [v[0] for v in values]
                refs = [v[1] for v in values]

                # Accuracy
                if len(refs) > 0:
                    acc = np.mean(np.array(preds) == np.array(refs))
                    acc_single_intr[interval] = acc

                # F1
                if len(set(refs)) > 1:
                    f1_result = f1_metric.compute(references=refs, predictions=preds, average='macro')
                    acc_single_intr_f1[interval] = f1_result['f1']
                else:
                    acc_single_intr_f1[interval] = 0.0

        # Save results
        results = {
            'model': 'tnt',
            'fold': FOLD,
            'val_composite_accuracy': composite_accuracy,
            'val_composite_micro_f1': composite_micro_f1,
            'val_composite_macro_f1': composite_macro_f1,
            'per_word_class_accuracy': {name_word_class[key]: acc_single_WT.get(key, 0.0) for key in name_word_class.keys()},
            'per_word_class_f1': {name_word_class[key]: acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()},
            'per_interval_accuracy': {name_intervals[key]: acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
            'per_interval_f1': {name_intervals[key]: acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
        }

        results_path = os.path.join(checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {results_path}")
        print(f"\n{'='*80}\n")

        # OOD evaluation for TnT
        if args.evaluate_ood and os.path.exists(args.ood_data_path):
            print(f"\n{'='*80}")
            print("EVALUATING ON OOD TEST SET")
            print(f"{'='*80}\n")

            # Load OOD data
            print(f"Loading OOD data from {args.ood_data_path}...")
            ood_sentences, ood_tags = load_ood_data(args.ood_data_path, tag_to_features)
            print(f"[OK] Loaded {len(ood_sentences)} sentences")

            # Convert OOD data to TnT format
            ood_tnt = prepare_tnt_data(ood_sentences, ood_tags, features_to_tag)
            print(f"[OK] Data converted - {len(ood_tnt)} sentences")

            # Evaluate
            all_pred_tags = []
            all_true_tags = []
            preds_WT = []
            refs_WT = []
            acc_single_WT = defaultdict(list)

            for sent_tnt in tqdm(ood_tnt, desc="Evaluating OOD"):
                words = [word for word, tag in sent_tnt]
                true_tags = [tag for word, tag in sent_tnt]

                # Predict
                pred_sent = tagger.tag(words)
                pred_tags = [tag for word, tag in pred_sent]

                # Collect predictions
                for pred_tag, true_tag in zip(pred_tags, true_tags):
                    all_pred_tags.append(pred_tag)
                    all_true_tags.append(true_tag)

                    # Extract word types
                    if pred_tag in tag_to_features and true_tag in tag_to_features:
                        pred_features = tag_to_features[pred_tag]
                        true_features = tag_to_features[true_tag]

                        pred_word_type = np.where(pred_features == 1)[0][0]
                        true_word_type = np.where(true_features == 1)[0][0]

                        preds_WT.append(pred_word_type)
                        refs_WT.append(true_word_type)
                        acc_single_WT[true_word_type].append((pred_word_type, true_word_type))

            # Compute metrics
            print(f"\n{'='*80}")
            print("OOD TEST RESULTS")
            print(f"{'='*80}\n")

            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load('f1')

            # Composite tag metrics
            composite_accuracy = np.mean(np.array(all_pred_tags) == np.array(all_true_tags))

            # Only keep pairs where both pred and true tags are in tag_to_id
            pred_tag_ids = []
            true_tag_ids = []
            for pred_tag, true_tag in zip(all_pred_tags, all_true_tags):
                if pred_tag in tag_to_id and true_tag in tag_to_id:
                    pred_tag_ids.append(tag_to_id[pred_tag])
                    true_tag_ids.append(tag_to_id[true_tag])

            composite_micro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='micro')['f1']
            composite_macro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='macro')['f1']

            print(f"Composite Tag Metrics:")
            print(f"  Accuracy:  {composite_accuracy:.4f}")
            print(f"  Micro F1:  {composite_micro_f1:.4f}")
            print(f"  Macro F1:  {composite_macro_f1:.4f}\n")

            # Word class metrics
            if len(preds_WT) > 0:
                accuracy_WT = accuracy_metric.compute(references=refs_WT, predictions=preds_WT)['accuracy']

                # Exclude outlier classes with very few examples from macro F1
                excluded_classes = [9, 11]  # Unanalyzed word, Web email or address
                filtered_refs_WT = []
                filtered_preds_WT = []
                for ref, pred in zip(refs_WT, preds_WT):
                    if ref not in excluded_classes:
                        filtered_refs_WT.append(ref)
                        filtered_preds_WT.append(pred)

                f1_WT = f1_metric.compute(references=filtered_refs_WT, predictions=filtered_preds_WT, average='macro')['f1'] if len(filtered_refs_WT) > 0 else 0.0

                print(f"Word Class Metrics:")
                print(f"  Accuracy:  {accuracy_WT:.4f}")
                print(f"  Macro F1:  {f1_WT:.4f}")

                # Calculate per-word-class accuracies
                for key in acc_single_WT:
                    correct_preds = sum(1 for pred, ref in acc_single_WT[key] if pred == ref)
                    acc = correct_preds / len(acc_single_WT[key])
                    acc_single_WT[key] = acc

                # Calculate per-word-class F1 scores
                unique_labels = sorted(set(refs_WT + preds_WT))
                f1_result = f1_metric.compute(references=refs_WT, predictions=preds_WT, average=None, labels=unique_labels)
                class_f1_scores = dict(zip(unique_labels, f1_result['f1']))
                acc_single_WT_f1 = {label: class_f1_scores.get(label, 0.0) for label in range(15)}

                print(f"\n  Per-Word-Class Accuracy:")
                for key in sorted(acc_single_WT.keys()):
                    if key in name_word_class:
                        print(f"    {name_word_class[key]:30s}: {acc_single_WT[key]:.4f}")

                print(f"\n  Per-Word-Class F1:")
                for key in sorted(acc_single_WT_f1.keys()):
                    if acc_single_WT_f1[key] > 0 and key in name_word_class:
                        print(f"    {name_word_class[key]:30s}: {acc_single_WT_f1[key]:.4f}")
                print()

            # Save OOD results
            ood_results = {
                'model': 'tnt',
                'fold': FOLD,
                'ood_data_path': args.ood_data_path,
                'num_sentences': len(ood_sentences),
                'composite_accuracy': composite_accuracy,
                'composite_micro_f1': composite_micro_f1,
                'composite_macro_f1': composite_macro_f1,
                'word_class_accuracy': accuracy_WT if len(preds_WT) > 0 else 0.0,
                'word_class_macro_f1': f1_WT if len(preds_WT) > 0 else 0.0,
                'per_word_class_accuracy': {name_word_class[key]: acc_single_WT.get(key, 0.0) for key in name_word_class.keys()} if len(preds_WT) > 0 else {},
                'per_word_class_f1': {name_word_class[key]: acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()} if len(preds_WT) > 0 else {},
            }

            # TnT doesn't use decoders, so no suffix needed
            ood_results_path = os.path.join(checkpoint_dir, 'ood_results.json')
            with open(ood_results_path, 'w') as f:
                json.dump(ood_results, f, indent=2)
            print(f"[OK] OOD results saved to: {ood_results_path}\n")

    # ============================================================================
    # NEURAL NETWORK MODEL
    # ============================================================================
    else:
        print(f"\nInitializing model ({MODE} mode)...")
        tokenizer, device, model = initialize_model(tag_to_features, mode=MODE, num_composite_tags=num_composite_tags)
        print(f"[OK] Model loaded on {device}")

        # Load TnT model for hybrid decoder if needed
        tnt_model = None
        tnt_log_A = None
        tnt_log_pi = None
        if args.eval_decoder == 'hybrid':
            from nltk.tag import tnt
            print(f"\n[Hybrid] Loading TnT model...")

            # Auto-detect TnT model path if not specified
            if args.tnt_model_path is None:
                tnt_path = os.path.join(args.output_dir, f'fold_{FOLD}_tnt/tnt_model.pkl')
            else:
                tnt_path = args.tnt_model_path

            if os.path.exists(tnt_path):
                with open(tnt_path, 'rb') as f:
                    tnt_model = pickle.load(f)
                print(f"[OK] TnT model loaded from {tnt_path}")

                # Extract transition matrix and unigram prior
                tnt_log_A, tnt_log_pi = extract_tnt_statistics(tnt_model, tag_to_features)
                tnt_log_A = tnt_log_A.to(device)
                tnt_log_pi = tnt_log_pi.to(device)
                print(f"[OK] Transition matrix extracted: {tnt_log_A.shape}")
                print(f"[OK] Unigram prior extracted: {tnt_log_pi.shape}")
            else:
                raise ValueError(f"TnT model not found: {tnt_path}. Train TnT first or specify --tnt_model_path")

        # Create checkpoint directory (needed for both training and eval-only)
        if MODE == 'multilabel' and args.unconstrained_loss:
            mode_suffix = f'multilabel_unconstrained_{args.unconstrained_loss}'
        else:
            mode_suffix = MODE
        if args.evaluate_only:
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            best_checkpoint_path = args.checkpoint_path
        else:
            checkpoint_dir = os.path.join(args.output_dir, f'fold_{FOLD}_{mode_suffix}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
        # Prepare validation loader for eval-only or training
        valid_dataset = CustomDataset(
            val_df['sentences'].tolist(),
            val_df['tags'].tolist(),
            tokenizer,
            MAX_LEN,
            mode=MODE,
            tag_to_id=tag_to_id,
            features_to_tag=features_to_tag
        )
        valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
        if not args.evaluate_only:
            train_dataset = CustomDataset(
                train_df['sentences'].tolist(),
                train_df['tags'].tolist(),
                tokenizer,
                MAX_LEN,
                mode=MODE,
                tag_to_id=tag_to_id,
                features_to_tag=features_to_tag
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize optimizer
            if OPTIMIZER == 'adafactor':
                # Adafactor with adaptive learning rate
                optimizer = Adafactor(
                    model.parameters(),
                    scale_parameter=True,
                    relative_step=True,
                    warmup_init=True,
                    lr=None
                )
                scheduler = None
                print(f"[OK] Using Adafactor optimizer with adaptive learning rate")
            elif OPTIMIZER == 'adam':
                # Adam with fixed learning rate
                optimizer = Adam(
                    model.parameters(),
                    lr=LEARNING_RATE
                )

                # Calculate total training steps and warmup steps
                total_steps = len(train_loader) * NUM_EPOCHS

                if WARMUP_STEPS > 0:
                    warmup_steps = WARMUP_STEPS
                    warmup_pct = (warmup_steps / total_steps) * 100 if total_steps > 0 else 0
                elif WARMUP_RATIO > 0:
                    warmup_steps = int(total_steps * WARMUP_RATIO)
                    warmup_pct = WARMUP_RATIO * 100
                else:
                    warmup_steps = 0
                    warmup_pct = 0

                # Linear warmup and decay scheduler
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                print(f"[OK] Using Adam optimizer with lr={LEARNING_RATE}")
                if warmup_steps > 0:
                    print(f"[OK] Linear scheduler: {warmup_steps} warmup steps ({warmup_pct:.1f}%), {total_steps} total steps")
                else:
                    print(f"[OK] Linear scheduler: no warmup, {total_steps} total steps")
            else:  # adamw
                # AdamW with fixed learning rate
                optimizer = AdamW(
                    model.parameters(),
                    lr=LEARNING_RATE
                )

                # Calculate total training steps and warmup steps
                total_steps = len(train_loader) * NUM_EPOCHS

                if WARMUP_STEPS > 0:
                    warmup_steps = WARMUP_STEPS
                    warmup_pct = (warmup_steps / total_steps) * 100 if total_steps > 0 else 0
                elif WARMUP_RATIO > 0:
                    warmup_steps = int(total_steps * WARMUP_RATIO)
                    warmup_pct = WARMUP_RATIO * 100
                else:
                    warmup_steps = 0
                    warmup_pct = 0

                # Linear warmup and decay scheduler
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                print(f"[OK] Using AdamW optimizer with lr={LEARNING_RATE}")
                if warmup_steps > 0:
                    print(f"[OK] Linear scheduler: {warmup_steps} warmup steps ({warmup_pct:.1f}%), {total_steps} total steps")
                else:
                    print(f"[OK] Linear scheduler: no warmup, {total_steps} total steps")
    
            # Early stopping variables
            best_micro_f1 = 0.0
            patience_counter = 0
            best_epoch = 0
            total_epochs_trained = 0
    
        all_word_classes = list(range(15))  # Assuming word class labels are from 0 to 14
    
        # Load checkpoint if in evaluation-only mode
        if args.evaluate_only:
            print("Loading checkpoint...")
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"[OK] Checkpoint loaded from {best_checkpoint_path}")
            if MODE == 'multilabel':
                if args.eval_decoder == 'mbr':
                    print(f"[OK] Using MBR decoder (temperature={args.temperature}, threshold={args.mbr_threshold})")
                elif args.eval_decoder == 'hybrid':
                    print(f"[OK] Using HYBRID decoder (T={args.hybrid_temperature}, α={args.hybrid_alpha}, β={args.hybrid_beta}, λ={args.hybrid_lambda}" +
                          (f", entropy_gate={args.hybrid_entropy_gate}" if args.hybrid_entropy_gate else "") + ")")
                else:
                    print(f"[OK] Using GREEDY decoder")
            print()

            # Evaluate on validation data in evaluate-only mode
            print(f"\n{'='*80}")
            print("EVALUATING ON VALIDATION DATA")
            print(f"{'='*80}\n")

            val_metrics = evaluate_model(
                model, valid_loader, device, MODE, tag_to_id, id_to_tag, tag_to_features,
                features_to_tag, intervals, dict_intervals, name_intervals, all_word_classes,
                eval_decoder=args.eval_decoder, temperature=args.temperature,
                mbr_threshold=args.mbr_threshold, mbr_weights=mbr_weights,
                interval_to_name=interval_to_name, description="Validation"
            )

            # Print results
            print(f"\n{'='*80}")
            print("VALIDATION RESULTS")
            print(f"{'='*80}\n")
            print(f"Composite Tag Metrics:")
            print(f"  Accuracy:  {val_metrics['composite_accuracy']:.4f}")
            print(f"  Micro F1:  {val_metrics['composite_micro_f1']:.4f}")
            print(f"  Macro F1:  {val_metrics['composite_macro_f1']:.4f}\n")
            print(f"Word Class Metrics:")
            print(f"  Accuracy:  {val_metrics['word_class_accuracy']:.4f}")
            print(f"  Macro F1:  {val_metrics['word_class_macro_f1']:.4f}\n")

            # Print per-word-class results
            print(f"  Per-Word-Class Accuracy:")
            for wc_id in sorted(val_metrics['per_word_class_accuracy'].keys()):
                wc_name = name_word_class.get(wc_id, f"WC_{wc_id}")
                print(f"    Word Class {wc_name:20s}: {val_metrics['per_word_class_accuracy'][wc_id]:.4f}")

            print(f"\n  Per-Word-Class F1:")
            for wc_id in sorted(val_metrics['per_word_class_f1'].keys()):
                wc_name = name_word_class.get(wc_id, f"WC_{wc_id}")
                print(f"    Word Class {wc_name:20s}: {val_metrics['per_word_class_f1'][wc_id]:.4f}")

            # Save results
            results_dict = {
                'mode': MODE,
                'fold': FOLD,
                'eval_decoder': args.eval_decoder,
                'val_composite_accuracy': val_metrics['composite_accuracy'],
                'val_composite_micro_f1': val_metrics['composite_micro_f1'],
                'val_composite_macro_f1': val_metrics['composite_macro_f1'],
                'word_class_accuracy': val_metrics['word_class_accuracy'],
                'word_class_macro_f1': val_metrics['word_class_macro_f1'],
                'per_word_class_accuracy': {name_word_class[k]: v for k, v in val_metrics['per_word_class_accuracy'].items() if k in name_word_class},
                'per_word_class_f1': {name_word_class[k]: v for k, v in val_metrics['per_word_class_f1'].items() if k in name_word_class},
                'per_interval_accuracy': val_metrics['per_interval_accuracy'],
                'per_interval_f1': val_metrics['per_interval_f1'],
            }

            results_path = os.path.join(checkpoint_dir, 'best_results.json')
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"\n[OK] Validation results saved to: {results_path}\n")

        # Training loop (skip if evaluate_only)
        if not args.evaluate_only:
            print(f"\n{'='*80}")
            print(f"Starting training (early stopping: patience={EARLY_STOPPING_PATIENCE}, delta={EARLY_STOPPING_DELTA})")
            if MODE == 'multilabel':
                if args.eval_decoder == 'mbr':
                    print(f"Eval decoder: MBR (temperature={args.temperature}, threshold={args.mbr_threshold})")
                else:
                    print(f"Eval decoder: GREEDY")
            print(f"{'='*80}\n")
    
            for epoch in range(NUM_EPOCHS):
                total_epochs_trained = epoch + 1
                print(f"\n{'-'*80}")
                print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
                print(f"{'-'*80}")
    
                model.train()
                train_loss = 0
    
                # Track train metrics
                train_pred_composite_tags = []
                train_true_composite_tags = []
    
                for batch in tqdm(train_loader, desc=f"  Training", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    begin_tokens = batch["begin_token"].to(device)
    
                    # MODE-SPECIFIC LOSS CALCULATION
                    if MODE == 'singlelabel':
                        # Single-label mode: use built-in cross-entropy loss
                        outputs = model(input_ids, attention_mask, labels=labels)
                        loss = outputs.loss
                    else:
                        # Multi-label mode: use hierarchical loss (don't pass labels to model)
                        outputs = model(input_ids, attention_mask)
                        predictions = outputs.logits
                        total_valid_tokens = 0
                        loss = 0

                        for idx, sentence in enumerate(input_ids):
                            for widx, word in enumerate(sentence):
                                if begin_tokens[idx][widx] == 0:
                                    continue
                                true_word_type = batch["word_type"][idx][widx]
                                if true_word_type == -9:
                                    continue

                                # TRAINING: compute loss on relevant features
                                if args.unconstrained_loss:
                                    loss += get_unconstrained_prediction(predictions[idx][widx], labels[idx][widx], intervals, device, normalize=(args.unconstrained_loss == 'normalized'))
                                else:
                                    loss += get_filtered_prediction(predictions[idx][widx], labels[idx][widx], dict_intervals, device, use_true_labels=True)
                                total_valid_tokens += 1
    
                        loss = loss / total_valid_tokens if total_valid_tokens > 0 else loss
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Step scheduler if using AdamW
                    if scheduler is not None:
                        scheduler.step()

                    # Track learning rate (if available)
                    if OPTIMIZER in ['adam', 'adamw']:
                        current_lr = optimizer.param_groups[0]['lr']
                    else:  # adafactor with relative_step
                        current_lr = optimizer.param_groups[0].get('lr', None)

                    train_loss += loss.item()
    
                    # Collect train composite tag predictions
                    with torch.no_grad():
                        if MODE == 'singlelabel':
                            pred_class_ids = torch.argmax(outputs.logits, dim=-1)
                            for idx in range(input_ids.size(0)):
                                for widx in range(input_ids.size(1)):
                                    if begin_tokens[idx][widx] == 1:
                                        pred_id = pred_class_ids[idx][widx].item()
                                        true_label = batch["original_labels"][idx][widx].cpu().numpy()
                                        if pred_id < len(id_to_tag):
                                            pred_tag = id_to_tag[pred_id]
                                            true_tag = vector_to_composite_tag(true_label, features_to_tag)
                                            if pred_tag and true_tag:
                                                train_pred_composite_tags.append(pred_tag)
                                                train_true_composite_tags.append(true_tag)
                        else:
                            # For multilabel, use the same prediction logic as validation
                            for idx in range(input_ids.size(0)):
                                predicted_labels = predict_classes(outputs.logits[idx], attention_mask[idx], begin_tokens[idx], dict_intervals)
    
                                # Get true labels for begin tokens (matching predict_classes behavior)
                                true_labels_for_sentence = []
                                for widx in range(labels.size(1)):
                                    if begin_tokens[idx][widx] == 1:
                                        true_label = labels[idx][widx].cpu().numpy()
                                        true_labels_for_sentence.append(true_label)
    
                                # Match predictions with true labels
                                for pred, true in zip(predicted_labels, true_labels_for_sentence):
                                    if len(pred) > 1:  # Only count valid predictions
                                        pred_tag = vector_to_composite_tag(pred, features_to_tag)
                                        true_tag = vector_to_composite_tag(true, features_to_tag)
                                        if pred_tag and true_tag:
                                            train_pred_composite_tags.append(pred_tag)
                                            train_true_composite_tags.append(true_tag)
    
                avg_loss_train = train_loss / len(train_loader)
    
                # Calculate train composite metrics
                train_composite_acc = 0.0
                train_composite_micro_f1 = 0.0
                train_composite_macro_f1 = 0.0
                if len(train_pred_composite_tags) > 0 and len(train_true_composite_tags) > 0:
                    train_composite_acc = np.mean(np.array(train_pred_composite_tags) == np.array(train_true_composite_tags))
                    train_pred_tag_ids = [tag_to_id[tag] for tag in train_pred_composite_tags if tag in tag_to_id]
                    train_true_tag_ids = [tag_to_id[tag] for tag in train_true_composite_tags if tag in tag_to_id]
                    if len(train_pred_tag_ids) > 0 and len(train_true_tag_ids) > 0:
                        f1_metric_train = evaluate.load('f1')
                        train_composite_micro_f1 = f1_metric_train.compute(references=train_true_tag_ids, predictions=train_pred_tag_ids, average='micro')['f1']
                        train_composite_macro_f1 = f1_metric_train.compute(references=train_true_tag_ids, predictions=train_pred_tag_ids, average='macro')['f1']
    
                # Validation - use unified evaluation function
                val_metrics = evaluate_model(
                    model, valid_loader, device, MODE, tag_to_id, id_to_tag, tag_to_features,
                    features_to_tag, intervals, dict_intervals, name_intervals, all_word_classes,
                    eval_decoder=args.eval_decoder, temperature=args.temperature,
                    mbr_threshold=args.mbr_threshold, mbr_weights=mbr_weights,
                    interval_to_name=interval_to_name, description="Validation"
                )

                # Extract metrics for compatibility with existing logging code
                avg_loss_valid = val_metrics['loss']
                composite_accuracy = val_metrics['composite_accuracy']
                composite_micro_f1 = val_metrics['composite_micro_f1']
                composite_macro_f1 = val_metrics['composite_macro_f1']
                accuracy_WT = {'accuracy': val_metrics['word_class_accuracy']}
                f1_WT = val_metrics['word_class_macro_f1']
                acc_single_WT = val_metrics['per_word_class_accuracy']
                acc_single_WT_f1 = val_metrics['per_word_class_f1']

                # Convert named intervals back to tuple keys for wandb logging
                acc_single_intr = {}
                acc_single_intr_f1 = {}
                for key, name in name_intervals.items():
                    acc_single_intr[key] = val_metrics['per_interval_accuracy'].get(name, 0.0)
                    acc_single_intr_f1[key] = val_metrics['per_interval_f1'].get(name, 0.0)

                sorted_acc_single_WT = dict(sorted(acc_single_WT.items()))
    
                # Console logging
                print(f"\n  Results:")
                print(f"    Train Loss: {avg_loss_train:.4f} | Val Loss: {avg_loss_valid:.4f}")
                print(f"    Train Composite - Acc: {train_composite_acc:.4f}, Micro F1: {train_composite_micro_f1:.4f}, Macro F1: {train_composite_macro_f1:.4f}")
                print(f"    Val Composite   - Acc: {composite_accuracy:.4f}, Micro F1: {composite_micro_f1:.4f} [EARLY STOP], Macro F1: {composite_macro_f1:.4f}")

                if current_lr is not None:
                    print(f"    Learning Rate: {current_lr:.2e} ({OPTIMIZER})")
                else:
                    print(f"    Learning Rate: adaptive ({OPTIMIZER})")
    
                # Prepare and log metrics to wandb
                other_log_data = {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss_train,
                    "val_loss": avg_loss_valid,
                    "train_composite_accuracy": train_composite_acc,
                    "train_composite_micro_f1": train_composite_micro_f1,
                    "train_composite_macro_f1": train_composite_macro_f1,
                    "val_composite_accuracy": composite_accuracy,
                    "val_composite_micro_f1": composite_micro_f1,
                    "val_composite_macro_f1": composite_macro_f1,
                    "eval_decoder": args.eval_decoder,
                    "mbr_threshold": args.mbr_threshold,
                }

                # Only add learning rate if available
                if current_lr is not None:
                    other_log_data["learning_rate"] = current_lr
    
                combined_log_data = {
                    **{f"accuracy_{name_word_class[key]}": acc_single_WT.get(key, 0.0) for key in name_word_class.keys()},
                    **{f"f1_{name_word_class[key]}": acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()},
                    **{f"accuracy_{name_intervals[key]}": acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
                    **{f"f1_{name_intervals[key]}": acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
                    **other_log_data
                }
    
                if HAS_WANDB:
                    wandb.log(combined_log_data)
    
                # Save per-epoch results JSON
                epoch_results = {
                    'epoch': epoch + 1,
                    'mode': MODE,
                    'fold': FOLD,
                    'eval_decoder': args.eval_decoder,
                    'mbr_threshold': args.mbr_threshold,
                    'train_loss': avg_loss_train,
                    'val_loss': avg_loss_valid,
                    'train_composite_accuracy': train_composite_acc,
                    'train_composite_micro_f1': train_composite_micro_f1,
                    'train_composite_macro_f1': train_composite_macro_f1,
                    'val_composite_accuracy': composite_accuracy,
                    'val_composite_micro_f1': composite_micro_f1,
                    'val_composite_macro_f1': composite_macro_f1,
                    'per_word_class_accuracy': {name_word_class[key]: acc_single_WT.get(key, 0.0) for key in name_word_class.keys()},
                    'per_word_class_f1': {name_word_class[key]: acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()},
                    'per_interval_accuracy': {name_intervals[key]: acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
                    'per_interval_f1': {name_intervals[key]: acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
                }

                # Add learning rate if available
                if current_lr is not None:
                    epoch_results['learning_rate'] = current_lr
    
                epoch_results_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_results.json')
                with open(epoch_results_path, 'w') as f:
                    json.dump(epoch_results, f, indent=2)
    
                # EARLY STOPPING AND CHECKPOINTING
                if composite_micro_f1 > best_micro_f1 + EARLY_STOPPING_DELTA:
                    best_micro_f1 = composite_micro_f1
                    patience_counter = 0
                    best_epoch = epoch + 1
    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_micro_f1': best_micro_f1,
                        'composite_accuracy': composite_accuracy,
                    }, best_checkpoint_path)
    
                    # Save results as JSON
                    results = {
                        'best_epoch': best_epoch,
                        'total_epochs_trained': total_epochs_trained,
                        'mode': MODE,
                        'fold': FOLD,
                        'optimizer': OPTIMIZER,
                        'eval_decoder': args.eval_decoder,
                        'mbr_threshold': args.mbr_threshold,
                        'train_loss': avg_loss_train,
                        'val_loss': avg_loss_valid,
                        'train_composite_accuracy': train_composite_acc,
                        'train_composite_micro_f1': train_composite_micro_f1,
                        'train_composite_macro_f1': train_composite_macro_f1,
                        'val_composite_accuracy': composite_accuracy,
                        'val_composite_micro_f1': composite_micro_f1,
                        'val_composite_macro_f1': composite_macro_f1,
                        'per_word_class_accuracy': {name_word_class[key]: acc_single_WT.get(key, 0.0) for key in name_word_class.keys()},
                        'per_word_class_f1': {name_word_class[key]: acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()},
                        'per_interval_accuracy': {name_intervals[key]: acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
                        'per_interval_f1': {name_intervals[key]: acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
                    }

                    # Add learning rate if available
                    if current_lr is not None:
                        results['learning_rate'] = current_lr
                    if OPTIMIZER in ['adam', 'adamw']:
                        results['base_learning_rate'] = LEARNING_RATE
                        results['warmup_ratio'] = WARMUP_RATIO
    
                    results_path = os.path.join(checkpoint_dir, 'best_results.json')
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
    
                    print(f'\n  >> NEW BEST MODEL! Micro F1: {best_micro_f1:.4f}')
                    print(f'     Saved to: {best_checkpoint_path}')
                    print(f'     Results saved to: {results_path}')
                else:
                    patience_counter += 1
                    print(f'\n  >> No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}')
    
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f'\n{"="*80}')
                    print(f'EARLY STOPPING TRIGGERED')
                    print(f'Best Micro F1: {best_micro_f1:.4f} (Epoch {epoch + 1 - patience_counter})')
                    print(f'{"="*80}\n')
                    break
    
            print(f"\n{'='*80}")
            print(f"[DONE] Training complete!")
            print(f"  Final model saved to: results_{FOLD}.pth")
            print(f"  Best checkpoint: {best_checkpoint_path}")
            print(f"{'='*80}\n")

            # Save in HuggingFace format if requested
            if args.save_huggingface:
                hf_dir = args.save_huggingface
                print(f"Saving HuggingFace model to {hf_dir}...")
                os.makedirs(hf_dir, exist_ok=True)

                # Load best checkpoint
                ckpt = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)

                # Save features_to_tag mapping (feature vector string -> tag)
                tag_mapping_data = {
                    ','.join(str(int(x)) for x in feats): tag
                    for tag, feats in tag_to_features.items()
                }
                with open(os.path.join(hf_dir, 'tag_mappings.json'), 'w') as f:
                    json.dump(tag_mapping_data, f, indent=2)

                # Save constraint mask (dict_intervals: word class -> active intervals)
                constraint_data = {
                    str(k): [list(iv) for iv in v]
                    for k, v in dict_intervals.items()
                }
                with open(os.path.join(hf_dir, 'constraint_mask.json'), 'w') as f:
                    json.dump(constraint_data, f, indent=2)

                print(f"[OK] HuggingFace model saved to {hf_dir}")

        # Evaluate on OOD data if requested
        if args.evaluate_ood and os.path.exists(args.ood_data_path):
            print(f"\n{'='*80}")
            print("EVALUATING ON OOD TEST SET")
            print(f"{'='*80}\n")
    
            # Load OOD data
            print(f"Loading OOD data from {args.ood_data_path}...")
            ood_sentences, ood_tags = load_ood_data(args.ood_data_path, tag_to_features)
            print(f"[OK] Loaded {len(ood_sentences)} sentences")
    
            # Create OOD dataset and dataloader
            ood_df = pd.DataFrame({'sentences': ood_sentences, 'tags': ood_tags})
            ood_dataset = CustomDataset(
                ood_df['sentences'].tolist(),
                ood_df['tags'].tolist(),
                tokenizer,
                MAX_LEN,
                mode=MODE,
                tag_to_id=tag_to_id,
                features_to_tag=features_to_tag
            )
            ood_loader = DataLoader(ood_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    
            # Load best model
            print("Loading best model...")
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("[OK] Best model loaded\n")
    
            # Evaluate
            preds_WT = []
            refs_WT = []
            acc_single_WT = defaultdict(list)
            acc_single_intr = defaultdict(list)
            all_pred_composite_tags = []
            all_true_composite_tags = []

            # For singlelabel per-interval metrics
            all_pred_tag_ids = []
            all_gold_tag_ids = []

            # Track sentence index for hybrid decoder
            sentence_idx = 0

            with torch.no_grad():
                for batch in tqdm(ood_loader, desc="Evaluating OOD"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    begin_tokens = batch["begin_token"].to(device)

                    if MODE == 'singlelabel':
                        outputs = model(input_ids, attention_mask)
                        predictions = outputs.logits
                        pred_class_ids = torch.argmax(predictions, dim=-1)

                        for idx in range(input_ids.size(0)):
                            for widx in range(input_ids.size(1)):
                                if begin_tokens[idx][widx] == 1:
                                    pred_id = pred_class_ids[idx][widx].item()
                                    true_label = batch["original_labels"][idx][widx].cpu().numpy()

                                    if pred_id < len(id_to_tag):
                                        pred_tag = id_to_tag[pred_id]
                                        true_tag = vector_to_composite_tag(true_label, features_to_tag)

                                        if pred_tag and true_tag:
                                            all_pred_composite_tags.append(pred_tag)
                                            all_true_composite_tags.append(true_tag)

                                            true_word_type = np.where(true_label == 1)[0][0]
                                            pred_features = tag_to_features[pred_tag]
                                            pred_word_type = np.where(pred_features == 1)[0][0]

                                            preds_WT.append(pred_word_type)
                                            refs_WT.append(true_word_type)
                                            acc_single_WT[true_word_type].append((pred_word_type, true_word_type))

                                            # Collect tag IDs for per-interval metrics
                                            true_tag_id = tag_to_id.get(true_tag, -1)
                                            if true_tag_id >= 0:
                                                all_pred_tag_ids.append(pred_id)
                                                all_gold_tag_ids.append(true_tag_id)
    
                    else:
                        outputs = model(input_ids, attention_mask)
                        predictions = outputs.logits

                        predictions_batch = []
                        for idx in range(input_ids.size(0)):
                            # Use MBR, hybrid, or greedy decoding based on flag
                            if args.eval_decoder == 'mbr':
                                # Apply temperature scaling for MBR
                                temp_scaled_logits = predictions[idx] / args.temperature
                                predicted_labels = mbr_decode_sentence(
                                    temp_scaled_logits, attention_mask[idx], begin_tokens[idx],
                                    dict_intervals, mbr_weights, interval_to_name, args.mbr_threshold
                                )
                            elif args.eval_decoder == 'hybrid':
                                # Get original tokens for this batch item using global sentence index
                                sent_tokens = ood_sentences[min(sentence_idx + idx, len(ood_sentences)-1)]

                                predicted_labels = hybrid_decode_sentence(
                                    predictions[idx], attention_mask[idx], begin_tokens[idx],
                                    sent_tokens, tnt_model, tag_to_features,
                                    dict_intervals, tnt_log_A, tnt_log_pi,
                                    args.hybrid_temperature, args.hybrid_alpha, args.hybrid_beta,
                                    args.hybrid_lambda, args.hybrid_entropy_gate,
                                    device
                                )
                            else:
                                predicted_labels = predict_classes(predictions[idx], attention_mask[idx], begin_tokens[idx], dict_intervals)
                            predictions_batch.append(predicted_labels)

                        # Diagnostic: count differences between MBR and greedy
                        if args.eval_decoder == 'mbr':
                            greedy_batch = []
                            for idx2 in range(input_ids.size(0)):
                                greedy_batch.append(
                                    predict_classes(predictions[idx2], attention_mask[idx2], begin_tokens[idx2], dict_intervals)
                                )

                            diff_tokens = 0
                            total_tokens = 0
                            for mbr_sent, grd_sent in zip(predictions_batch, greedy_batch):
                                for mbr_vec, grd_vec in zip(mbr_sent, grd_sent):
                                    total_tokens += 1
                                    if not torch.equal((mbr_vec > 0).to(torch.int), (grd_vec > 0).to(torch.int)):
                                        diff_tokens += 1
                            if diff_tokens > 0:
                                print(f"      [BATCH] MBR differs from greedy: {diff_tokens}/{total_tokens} tokens ({100*diff_tokens/total_tokens:.1f}%)")
    
                        preds_WT_batch, refs_WT_batch, _, accuracy_single_WT_batch, accuracy_single_interval = calculate_accuracy(
                            predictions_batch, labels, dict_intervals, device, begin_tokens
                        )
    
                        preds_WT.extend(preds_WT_batch)
                        refs_WT.extend(refs_WT_batch)
    
                        for key, value in accuracy_single_WT_batch.items():
                            acc_single_WT[key].extend(value)
    
                        for key, value in accuracy_single_interval.items():
                            acc_single_intr[key].extend(value)
    
                        # Get composite tags
                        for idx, pred_batch in enumerate(predictions_batch):
                            true_labels_for_sentence = []
                            for widx in range(labels.size(1)):
                                if begin_tokens[idx][widx] == 1:
                                    true_label = labels[idx][widx].cpu().numpy()
                                    true_labels_for_sentence.append(true_label)
    
                            for pred, true in zip(pred_batch, true_labels_for_sentence):
                                if len(pred) > 1:
                                    pred_tag = vector_to_composite_tag(pred, features_to_tag)
                                    true_tag = vector_to_composite_tag(true, features_to_tag)
                                    if pred_tag and true_tag:
                                        all_pred_composite_tags.append(pred_tag)
                                        all_true_composite_tags.append(true_tag)

                    # Increment sentence index for next batch
                    sentence_idx += input_ids.size(0)

            # Compute and print OOD metrics
            print(f"\n{'='*80}")
            print("OOD TEST RESULTS")
            print(f"{'='*80}\n")
    
            # Load metrics
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load('f1')
    
            # Composite metrics
            if len(all_pred_composite_tags) > 0:
                ood_composite_acc = np.mean(np.array(all_pred_composite_tags) == np.array(all_true_composite_tags))
                pred_tag_ids = [tag_to_id[tag] for tag in all_pred_composite_tags if tag in tag_to_id]
                true_tag_ids = [tag_to_id[tag] for tag in all_true_composite_tags if tag in tag_to_id]
    
                if len(pred_tag_ids) > 0:
                    ood_composite_micro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='micro')['f1']
                    ood_composite_macro_f1 = f1_metric.compute(references=true_tag_ids, predictions=pred_tag_ids, average='macro')['f1']
    
                    print(f"Composite Tag Metrics:")
                    print(f"  Accuracy:  {ood_composite_acc:.4f}")
                    print(f"  Micro F1:  {ood_composite_micro_f1:.4f}")
                    print(f"  Macro F1:  {ood_composite_macro_f1:.4f}\n")
    
            # Word class metrics
            if len(preds_WT) > 0:
                ood_wt_acc = accuracy_metric.compute(references=refs_WT, predictions=preds_WT)['accuracy']

                # Exclude outlier classes with very few examples from macro F1
                excluded_classes = [9, 11]  # Unanalyzed word, Web email or address
                filtered_refs_WT = []
                filtered_preds_WT = []
                for ref, pred in zip(refs_WT, preds_WT):
                    if ref not in excluded_classes:
                        filtered_refs_WT.append(ref)
                        filtered_preds_WT.append(pred)

                ood_wt_f1 = f1_metric.compute(references=filtered_refs_WT, predictions=filtered_preds_WT, average='macro')['f1'] if len(filtered_refs_WT) > 0 else 0.0

                print(f"Word Class Metrics:")
                print(f"  Accuracy:  {ood_wt_acc:.4f}")
                print(f"  Macro F1:  {ood_wt_f1:.4f}")

                # Calculate per-word-class accuracies
                for key in acc_single_WT:
                    correct_preds = sum(1 for pred, ref in acc_single_WT[key] if pred == ref)
                    acc = correct_preds / len(acc_single_WT[key])
                    acc_single_WT[key] = acc

                # Calculate per-word-class F1 scores
                unique_labels = sorted(set(refs_WT + preds_WT))
                f1_result = f1_metric.compute(references=refs_WT, predictions=preds_WT, average=None, labels=unique_labels)
                class_f1_scores = dict(zip(unique_labels, f1_result['f1']))
                acc_single_WT_f1 = {label: class_f1_scores.get(label, 0.0) for label in range(15)}

                print(f"\n  Per-Word-Class Accuracy:")
                for key in sorted(acc_single_WT.keys()):
                    if key in name_word_class:
                        print(f"    {name_word_class[key]:30s}: {acc_single_WT[key]:.4f}")

                print(f"\n  Per-Word-Class F1:")
                for key in sorted(acc_single_WT_f1.keys()):
                    if acc_single_WT_f1[key] > 0 and key in name_word_class:
                        print(f"    {name_word_class[key]:30s}: {acc_single_WT_f1[key]:.4f}")
                print()

            # Compute per-interval metrics for singlelabel mode
            acc_single_intr_f1 = {}
            if MODE == 'singlelabel' and len(all_pred_tag_ids) > 0 and len(all_gold_tag_ids) > 0:
                interval_metrics = calculate_accuracy_singlelabel(
                    all_pred_tag_ids, all_gold_tag_ids, id_to_tag, tag_to_features, intervals
                )

                # Calculate accuracy and F1 for each interval
                for interval, values in interval_metrics.items():
                    preds = [v[0] for v in values]
                    refs = [v[1] for v in values]

                    # Accuracy
                    if len(refs) > 0:
                        acc = np.mean(np.array(preds) == np.array(refs))
                        acc_single_intr[interval] = acc

                    # F1
                    if len(set(refs)) > 1:
                        f1_result = f1_metric.compute(references=refs, predictions=preds, average='macro')
                        acc_single_intr_f1[interval] = f1_result['f1']
                    else:
                        acc_single_intr_f1[interval] = 0.0
            elif MODE == 'multilabel':
                # For multilabel, compute F1 from already populated acc_single_intr
                for key in acc_single_intr.keys():
                    values = acc_single_intr[key]
                    preds = [tup[0] for tup in values]
                    refs = [tup[1] for tup in values]
                    if len(set(refs)) > 1:
                        f1_result = f1_metric.compute(references=refs, predictions=preds, average='macro')
                        acc_single_intr_f1[key] = f1_result['f1']
                    else:
                        acc_single_intr_f1[key] = 0.0

            # Save OOD results
            ood_results = {
                'mode': MODE,
                'fold': FOLD,
                'eval_decoder': args.eval_decoder,
                'mbr_threshold': args.mbr_threshold,
                'temperature': args.temperature,
                'ood_data_path': args.ood_data_path,
                'num_sentences': len(ood_sentences),
                'composite_accuracy': ood_composite_acc if len(all_pred_composite_tags) > 0 else 0.0,
                'composite_micro_f1': ood_composite_micro_f1 if len(pred_tag_ids) > 0 else 0.0,
                'composite_macro_f1': ood_composite_macro_f1 if len(pred_tag_ids) > 0 else 0.0,
                'word_class_accuracy': ood_wt_acc if len(preds_WT) > 0 else 0.0,
                'word_class_macro_f1': ood_wt_f1 if len(preds_WT) > 0 else 0.0,
                'per_word_class_accuracy': {name_word_class[key]: acc_single_WT.get(key, 0.0) for key in name_word_class.keys()} if len(preds_WT) > 0 else {},
                'per_word_class_f1': {name_word_class[key]: acc_single_WT_f1.get(key, 0.0) for key in name_word_class.keys()} if len(preds_WT) > 0 else {},
                'per_interval_accuracy': {name_intervals.get(key, str(key)): acc_single_intr.get(key, 0.0) for key in name_intervals.keys()},
                'per_interval_f1': {name_intervals.get(key, str(key)): acc_single_intr_f1.get(key, 0.0) for key in name_intervals.keys()},
            }

            # Create decoder-specific filename to avoid overwriting results
            if args.eval_decoder == 'mbr':
                temp_str = f"_T{args.temperature:.1f}".replace('.', 'p') if args.temperature != 1.0 else ""
                thresh_str = f"_t{args.mbr_threshold:.2f}".replace('.', 'p')
                decoder_suffix = f"_mbr{temp_str}{thresh_str}"
            else:
                decoder_suffix = ""

            ood_results_path = os.path.join(checkpoint_dir, f'ood_results{decoder_suffix}.json')
            with open(ood_results_path, 'w') as f:
                json.dump(ood_results, f, indent=2)
            print(f"[OK] OOD results saved to: {ood_results_path}\n")
    
        # Only finish wandb if we initialized it (i.e., not in evaluation-only mode)
        if not args.evaluate_only:
            if HAS_WANDB:
                wandb.finish()

if __name__ == "__main__":
    main()
