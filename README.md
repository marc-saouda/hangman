# Hangman: Transformer-Based Letter Prediction

A deep learning approach to playing Hangman, achieving a **56% win rate** on an unseen 250,000-word test set (up from an 18% baseline). The model uses a Transformer encoder with BPE tokenization and a supervised learning pipeline to predict masked letters from partially revealed words.

## Repository Structure

```
├── notebooks/
│   ├── hangman_api_user.ipynb        # Game client — plays 1,000 games against the API
│   ├── Transformer.ipynb             # Model training (Google Colab)
│   ├── dataset_construction.ipynb    # Supervised dataset generation
│   ├── tokenizer.ipynb               # BPE tokenizer construction
│   ├── fine_tuning_dataset.ipynb     # Fine-tuning dataset + LoRA experiments
│   └── LSTM.ipynb                    # Earlier LSTM baseline (superseded)
├── models/
│   ├── 10_epochs.pt                  # Final trained checkpoint (used at inference)
│   ├── 5_epochs.pt                   # Intermediate checkpoint
│   ├── 15_epochs_finetuned.pt        # Fine-tuned checkpoint
│   └── lora_finetuned.pt             # LoRA adapter weights
├── data/
│   ├── words_250000_train.txt        # Training dictionary (250K words)
│   ├── tokenizer.json                # BPE vocabulary (100 merges, used at inference)
│   ├── tokenizer_{150,200,500,750,2500}.json  # Alternative tokenizers
│   └── *.csv                         # Fine-tuning datasets
├── README.md
└── README.TXT                        # Detailed design notes
```

## Architecture

```
Masked Word → BPE Tokenizer → Token Embeddings + Positional Encoding
    → Transformer Encoder (4 layers, 4 heads, d=256)
    → Adaptive Avg Pooling
    → Concat with Wrong-Guesses Embedding (26→256)
    → Classifier (512→256→26) → Letter Probabilities
```

**Key design choices:**
- **BPE tokenization** (100 merges) captures subword patterns (e.g., common prefixes/suffixes), giving the model richer input representations than character-level encoding
- **Wrong-guesses vector** — a 26-dim one-hot input lets the model condition predictions on previously failed letters
- **Fallback strategy** — for 3- and 4-letter words, frequency-based dictionary lookup outperforms the model and is used instead

## Dataset Construction

For each word in the 250K training dictionary:

1. **Enumerate** all non-empty subsets of the word's distinct letters
2. **Mask** every occurrence of the chosen subset with `_`
3. **Label** the example with the most-frequent masked letter (by English frequency order `etaoinshrdlcumwfgypbvkjxqz`)
4. **Compute** a wrong-guesses vector by marking the first 6 frequency-ordered letters absent from the word

This produces ~20M training examples that teach the model to recover letters from partial word observations.

## Training

- **Optimizer:** AdamW, lr = 5e-5
- **Loss:** BCEWithLogitsLoss (multi-label, since multiple letters can be correct)
- **Epochs:** 10 total (5 initial + 5 continued) on ~17M train / ~2M val split
- **Hardware:** Google Colab GPU (NVIDIA T4 / A100)

## Experiments & Observations

| Approach | Outcome |
|---|---|
| LSTM baseline | Decent starting point, but limited context window |
| Transformer (5 epochs) | ~55.6% win rate |
| Transformer (10 epochs) | **56% win rate** (final) |
| BPE merges (100–2000) | All converged to similar performance; 100 chosen for efficiency |
| More attention heads (4 to 8) | No improvement — model appeared capacity-saturated |
| LoRA fine-tuning | No measurable gain under compute constraints |
| Char-level CNN front-end | Performance unchanged |
| English-only training data | Degraded win rate — multilingual diversity helps generalization |

**Core bottleneck:** data scarcity. All architectural changes plateaued quickly, pointing to dataset size as the primary lever for further gains.

## Quickstart

```bash
pip install torch numpy requests
```

1. Open `notebooks/hangman_api_user.ipynb`
2. Ensure `data/tokenizer.json` and `models/10_epochs.pt` are present
3. Set your API access token in the `HangmanAPI` constructor
4. Run all cells — the notebook loads the model, connects to the API, and plays games

## Future Work

- **Synthetic data augmentation** — morphological derivations, noisy spellings, LM-guided mask completion
- **Longer training** on expanded datasets (15–20 epochs with early stopping)
- **Hyperparameter search** — learning rate scheduling, BPE merge count, classifier depth
- **Architecture tweaks** — more layers/heads once data bottleneck is addressed
