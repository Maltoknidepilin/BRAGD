# BRAGD: POS Tagger for Faroese

Code and data for **"BRAGD: Constrained Multi-Label POS Tagging for Faroese"**, LREC-COLING 2026.

**Authors:** Annika Simonsen, Barbara Scalvini, Uni Johannesen, Iben Nyholm Debess, Hafsteinn Einarsson, Vésteinn Snæbjarnarson

The exact code used for the experiments reported in the paper is available in release [`v1.0`](https://github.com/Maltoknidepilin/BRAGD/releases/tag/v1.0). The `main` branch may contain updates and improvements made after publication.

## Overview

We propose a multi-label approach to POS tagging for Faroese, where each token's tag is decomposed into a 73-dimensional binary feature vector covering word class, subcategory, gender, number, case, and other morphological attributes. A constrained loss function restricts gradient updates to only the feature groups relevant to each token's word class, preventing spurious learning signals from irrelevant categories (e.g., tense for nouns).

The tagger fine-tunes [ScandiBERT](https://huggingface.co/vesteinn/ScandiBERT) and achieves 97.5% accuracy on the Sosialurin-BRAGD corpus (10-fold CV) and 96.2% on out-of-domain data.

## Resources

- **Model on Hugging Face:** [BRAGD](https://huggingface.co/Setur/BRAGD)
- **Interactive demo:** [Marka](https://huggingface.co/spaces/Setur/Marka)


## Installation 
### Choose your shell

This repository supports:
- Linux/macOS: shell commands as written below
- Windows PowerShell: recommended for setup and inference
- Windows Git Bash: use this for running the shell scripts in `scripts/`

Depending on your system, you may need `python3` instead of `python`. On Windows PowerShell, `py -3` may also work.


### Clone the repository
```bash
git clone https://github.com/Maltoknidepilin/BRAGD.git
cd BRAGD
```
All commands below assume you are running the commands from the repository root.

### Create virtual environment
macOS/Linux/Windows:
```
python -m venv .venv
```
### Activate the virtual environment

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```PowerShell
.\.venv\Scripts\Activate.ps1
```

Windows Git Bash:
```bash
source .venv/Scripts/activate
```

### Install PyTorch (GPU optional)

If you want to use the GPU for training or inference, install a CUDA-enabled PyTorch build **before** installing the package below. If you are using CPU only, you can skip this step.

For example, for NVIDIA GPU + CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
If you are unsure which PyTorch build to install for your machine, see the official PyTorch installer:
https://pytorch.org/get-started/locally/

### Install the Package
*If you only want to run inference with the pretrained model, skip to “Quick Start: Inference Only”.*

Install the package for training, evaluation, and local development:

```
pip install -e .
```
#### Optional for wandb experiment logging: 
```
pip install -e ".[logging]"
```

## Quick Start: Inference Only
Tag Faroese text using the pre-trained [model from HuggingFace](https://huggingface.co/Setur/BRAGD). This release model is trained on both Sosialurin-BRAGD and OOD-BRAGD.

These examples assume you are running the commands from the repository root. 
If you only want to use the pretrained tagger and do not need the full training and evaluation setup, install only the inference dependencies below.

### Install requirements for *Inference Only*
If you already ran `pip install -e .`, skip this step.

```
pip install numpy torch "transformers==4.41.2" sentencepiece
```

Then run:
```
python inference.py "Hetta er eitt føroyskt dømi"
```

### In Python

```python
from inference import load_tagger

tagger = load_tagger("Setur/BRAGD")
results = tagger.tag("Hetta er eitt føroyskt dømi")

for token in results:
    print(f"{token['word']:15s} {token['tag']:10s} {token['features']}")
```
This returns one result per token, including the predicted BRAGD tag and its decoded feature values.

### Batched Inference (multiple sentences)
```python
from inference import load_tagger

tagger = load_tagger("Setur/BRAGD")
results = tagger.tag_sentences([
    ["Hetta", "er", "eitt", "føroyskt", "dømi"],
    ["Vit", "fara", "til", "Keypmannahavnar"],
])

for sentence in results:
    for token in sentence:
        print(f"{token['word']:15s} {token['tag']:10s} {token['features']}")
    print()
```

### Example Output
Each line shows the token, the predicted BRAGD tag, and the decoded morphological features. You can find an explanation of the tags in `data/BRAGD_tagset.md`.
```text
Hetta           PDNpSN     {'word_class': 'Pronoun', 'subcategory': 'D', 'gender': 'N', 'number': 'S', 'case': 'N', 'person': 'p'}
er              VNAPS3     {'word_class': 'Verb', 'number': 'S', 'mood': 'N', 'voice': 'A', 'tense': 'P', 'person': '3'}
eitt            RNSNI      {'word_class': 'Article', 'gender': 'N', 'number': 'S', 'case': 'N', 'definiteness': 'I'}
føroyskt        APSNSN     {'word_class': 'Adjective', 'gender': 'N', 'number': 'S', 'case': 'N', 'degree': 'P', 'declension': 'S'}
dømi            SNSNar     {'word_class': 'Noun', 'gender': 'N', 'number': 'S', 'case': 'N', 'article': 'a', 'proper_noun': 'r'}
```

## Reproducing Paper Results

Pre-computed results for all models are included in `results/`. 

### Regenerate tables and statistics

```
# Generate LaTeX tables and training progress figure
python generate_tables.py --skip-abltagger

# Run statistical significance tests
python compute_statistics.py
```

### Retrain all models from scratch (takes a few hours):
Windows: We recommend installing Git Bash and running this from there. Remember to go to root directory and activate environment first: `source .venv/Scripts/activate`

```bash
bash scripts/train_all_bragd.sh
```

This trains 5 model variants (TnT, single-label, multi-label constrained, unconstrained normalized, unconstrained unnormalized) across 10 folds. Results and logs are saved under `results/all_splits_eval_bragd_adamw/`.

## Training

### Train a release model on all available data

To train a single model on all available data (corpus + OOD) for deployment (macOS/Linux/Windows):
```bash
bash scripts/train_release_model.sh
```
Windows: We recommend installing Git Bash and running this from there. Remember to go to root directory and activate environment first: `source .venv/Scripts/activate`

This trains for 20 fixed epochs and saves a HuggingFace-compatible model to `release_model/huggingface/`. 

### Training Individual Models

```
# ScandiBERT multi-label (constrained loss), fold 0
python POS_tagger.py --mode multilabel --fold 0 --optimizer adamw --learning_rate 2e-5 --batch_size 8 --output_dir results/ --evaluate_ood

# ScandiBERT multi-label (unconstrained loss, normalized), fold 0
python POS_tagger.py --mode multilabel --unconstrained_loss normalized --fold 0 --optimizer adamw --learning_rate 2e-5 --batch_size 8 --output_dir results/ --evaluate_ood

# ScandiBERT single-label, fold 0
python POS_tagger.py --mode singlelabel --fold 0 --optimizer adamw --learning_rate 2e-5 --batch_size 8 --output_dir results/ --evaluate_ood

# TnT baseline (no GPU required)
python POS_tagger.py --mode singlelabel --model_type tnt --fold 0 --output_dir results/ --evaluate_ood
```

### Key CLI Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `singlelabel` or `multilabel` |
| `--model_type` | `neural` (default, ScandiBERT) or `tnt` |
| `--unconstrained_loss` | `unnormalized` or `normalized` (ablation, with `--mode multilabel`) |
| `--fold` | Cross-validation fold (0--9) |
| `--optimizer` | `adafactor` (default), `adam`, or `adamw` |
| `--learning_rate` | Learning rate (default: 2e-5) |
| `--batch_size` | Batch size (default: 32, paper uses 8) |
| `--evaluate_ood` | Run OOD evaluation after training |
| `--full_train` | Train on all data (no train/val split) |
| `--include_ood` | Include OOD data in training (with `--full_train`) |
| `--fixed_epochs N` | Train for exactly N epochs (no early stopping) |
| `--save_huggingface DIR` | Save model in HuggingFace format |


## Data

All data files are in `data/`:

| File | Description |
|------|-------------|
| `Sosialurin-BRAGD.tsv` | Annotated Faroese corpus (6,099 sentences, 123k tokens). Each line: `token\ttag`, sentences separated by `EOS\tEOS`. |
| `Sosialurin-BRAGD_tags.csv` | Tag-to-feature mapping (651 unique tags, 73 binary features). |
| `OOD-BRAGD.json` | Out-of-domain evaluation set (500 sentences from mixed genres). |
| `split_indices.npy` | Pre-computed 10-fold cross-validation splits (reproducible, seed=42). |
| `BRAGD_tagset.md` | Explanation of the tagset used for Sosialurin-BRAGD and OOD-BRAGD. | 

The corpus is a revised and re-annotated version of Hinrik Hafsteinsson's [Sosialurin Revised Corpus](https://github.com/hinrikur/FAR-GOLD/blob/master/correction/finished/sosialurin-revised.txt), using the BRAGD tagset (73 features across 15 word classes and 13 morphological feature groups). Hinrik Hafsteinsson's revised and re-annotated version is built on the tagged corpus created by Zakaris Svabo Hansen, Heini Justinussen, and Mortan Ólason (2004) "Marking av teldutøkum tekstsavni" (Tagging of a digital text corpus); for further details, see their [project page](https://studulsyvirlitid.gransking.fo/index.php?type=person&lng=en&id=18). See the paper for full tagset documentation.

### Feature structure (73 dimensions)

| Indices | Feature Group |
|---------|---------------|
| 0--14 | Word class (15 classes: Noun, Adjective, Pronoun, Number, Verb, Participle, Adverb, Conjunction, Foreign, Unanalyzed, Abbreviation, Web/Email, Punctuation, Symbol, Article) |
| 15--29 | Subcategory |
| 30--33 | Gender |
| 34--36 | Number |
| 37--41 | Case |
| 42--43 | Article |
| 44--45 | Proper noun |
| 46--50 | Degree |
| 51--53 | Declension |
| 54--60 | Mood |
| 61--63 | Voice |
| 64--66 | Tense |
| 67--70 | Person |
| 71--72 | Definiteness |

## Repository Structure

```
faroese-pos/
├── POS_tagger.py          # Training and evaluation
├── data_utils.py           # Data loading and tag mappings
├── inference.py            # Inference with pre-trained model
├── generate_tables.py      # LaTeX tables and figures for the paper
├── compute_statistics.py   # Statistical significance tests
├── make_splits.py          # Generate cross-validation splits
├── pyproject.toml          # Dependencies
├── LICENSE                 # MIT (code) + CC BY 4.0 (data)
├── scripts/
│   ├── train_all_bragd.sh      # Full 10-fold CV experiment
│   └── train_release_model.sh  # Train release model on all data
├── data/
│   ├── BRAGD_tagset.md             # Human-friendly tagset
│   ├── Sosialurin-BRAGD.tsv        # Corpus
│   ├── Sosialurin-BRAGD_tags.csv   # Tag definitions
│   ├── OOD-BRAGD.json              # OOD evaluation data
│   └── split_indices.npy           # CV splits
└── results/                        # Pre-computed 10-fold results
```

## Baselines

The ABLTagger baseline is in a separate repository: [far-ABLTagger](https://github.com/hinrikur/far-ABLTagger). To include ABLTagger results in table generation:

```bash
python generate_tables.py \
    --abltagger-val-summary /path/to/abltagger/summary.json \
    --abltagger-ood-summary /path/to/abltagger/ood_summary.json
```

## Citation

```bibtex
@inproceedings{simonsen2026bragd,
    title={{BRAGD}: Constrained Multi-Label {POS} Tagging for {F}aroese},
    author={Simonsen, Annika and Scalvini, Barbara and Johannesen, Uni and Debess, Iben Nyholm and Einarsson, Hafsteinn and Sn{\ae}bjarnarson, V{\'e}steinn},
    booktitle={Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
    year={2026}
}
```

## License

- **Code**: MIT License
- **Data** (corpus, tagset, OOD evaluation): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
