"""
Data loading and processing utilities for Faroese POS tagging
"""
import pandas as pd
import numpy as np
import json


def load_and_process_corpus(tags_filepath, corpus_filepath):
    """Load POS tags and corpus data.

    Args:
        tags_filepath: Path to CSV with tag features
        corpus_filepath: Path to TSV corpus file

    Returns:
        sentences: List of word lists
        sentence_tags: List of feature vector lists
        tag_to_features: Dict mapping tag strings to feature vectors
    """
    # Load POS tags and features
    tags_df = pd.read_csv(tags_filepath)
    tag_to_features = {row['Original Tag']: row[1:].values.astype(int) for _, row in tags_df.iterrows()}

    # Load the corpus
    with open(corpus_filepath, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    # Process the corpus into sentences and tags
    sentences, sentence_tags = [], []
    current_sentence, current_tags = [], []
    for line in corpus:
        if line.strip() == 'EOS\tEOS':
            if current_sentence:
                sentences.append(current_sentence)
                sentence_tags.append(current_tags)
                current_sentence, current_tags = [], []
        else:
            token, tag = line.strip().split('\t')
            if tag in tag_to_features:
                current_sentence.append(token)
                current_tags.append(tag_to_features[tag])
            else:
                print(f"Tag not found in tag_to_features: {tag}")
                continue

    return sentences, sentence_tags, tag_to_features


def load_ood_data(ood_json_path, tag_to_features):
    """Load and parse OOD data from JSON format.

    Args:
        ood_json_path: Path to JSON file with OOD data
        tag_to_features: Dict mapping tag strings to feature vectors

    Returns:
        sentences: List of word lists
        sentence_tags: List of feature vector lists
    """
    with open(ood_json_path, 'r', encoding='utf-8') as f:
        ood_data = json.load(f)

    sentences = []
    sentence_tags = []
    skipped = 0
    unknown_tags = set()

    for sent_obj in ood_data['sentences']:
        tokens = [t['token'] for t in sent_obj['tokens']]
        tags = [t['tag'] for t in sent_obj['tokens']]

        # Convert tags to feature vectors
        tag_features = []
        valid_sentence = True
        for tag in tags:
            if tag in tag_to_features:
                tag_features.append(tag_to_features[tag])
            else:
                unknown_tags.add(tag)
                valid_sentence = False
                break

        if valid_sentence:
            sentences.append(tokens)
            sentence_tags.append(tag_features)
        else:
            skipped += 1

    if unknown_tags:
        print(f"[WARNING] Skipped {skipped} sentences with unknown tags: {list(unknown_tags)[:5]}{'...' if len(unknown_tags) > 5 else ''}")

    return sentences, sentence_tags


def create_tag_mappings(tag_to_features):
    """Create mappings between feature vectors and composite tags.

    Args:
        tag_to_features: Dict mapping tag strings to feature vectors

    Returns:
        features_to_tag: Dict mapping feature tuples to tag strings
        tag_to_id: Dict mapping tag strings to integer IDs
        id_to_tag: Dict mapping integer IDs to tag strings
    """
    # Create reverse mapping: tuple of features -> composite tag
    features_to_tag = {tuple(features): tag for tag, features in tag_to_features.items()}

    # Create mappings for single-label mode
    unique_tags = sorted(tag_to_features.keys())
    tag_to_id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}

    return features_to_tag, tag_to_id, id_to_tag


def prepare_tnt_data(sentences, sentence_tags, features_to_tag):
    """Convert feature vectors to composite tags for TnT tagger.

    Args:
        sentences: List of word lists
        sentence_tags: List of feature vector lists
        features_to_tag: Dict mapping feature tuples to composite tags

    Returns:
        List of tagged sentences in format [[(word, tag), ...], ...]
    """
    tnt_data = []
    skipped = 0

    for sent, tags in zip(sentences, sentence_tags):
        tagged_sent = []
        valid_sentence = True

        for word, tag_vec in zip(sent, tags):
            composite_tag = features_to_tag.get(tuple(tag_vec))
            if composite_tag:
                tagged_sent.append((word, composite_tag))
            else:
                valid_sentence = False
                skipped += 1
                break

        if valid_sentence and len(tagged_sent) > 0:
            tnt_data.append(tagged_sent)

    if skipped > 0:
        print(f"[WARNING] Skipped {skipped} sentences with unmapped tags")

    return tnt_data


def load_split_indices(fold):
    """Load train/val split indices for a given fold.

    Args:
        fold: Fold number

    Returns:
        train_indexes: List of training sentence indices
        val_indexes: List of validation sentence indices
    """
    splits = np.load('data/split_indices.npy', allow_pickle=True).item()
    return splits[fold]['train'], splits[fold]['val']


def train_test_split_data(sentences, sentence_tags, train_indexes, val_indexes):
    """Split data into train and validation sets.

    Args:
        sentences: List of all sentences
        sentence_tags: List of all sentence tags
        train_indexes: Indices for training set
        val_indexes: Indices for validation set

    Returns:
        train_df: DataFrame with training data
        val_df: DataFrame with validation data
    """
    train_sentences = [sentences[i] for i in train_indexes]
    train_tags = [sentence_tags[i] for i in train_indexes]
    val_sentences = [sentences[i] for i in val_indexes]
    val_tags = [sentence_tags[i] for i in val_indexes]

    df_train = {'sentences': train_sentences, 'tags': train_tags}
    df_train = pd.DataFrame(df_train)
    df_val = {'sentences': val_sentences, 'tags': val_tags}
    df_val = pd.DataFrame(df_val)
    return df_train, df_val


def vector_to_composite_tag(feature_vector, features_to_tag):
    """Convert a feature vector to its composite tag string.

    Args:
        feature_vector: Feature vector (numpy array or tensor)
        features_to_tag: Dict mapping feature tuples to tags

    Returns:
        Composite tag string or None if not found
    """
    import torch
    if isinstance(feature_vector, torch.Tensor):
        feature_vector = feature_vector.cpu().numpy()
    feature_tuple = tuple(feature_vector.astype(int))
    return features_to_tag.get(feature_tuple, None)
