# Traditional Semantic Communication (PyTorch)

End-to-end implementation of a **traditional semantic communication system for text** using PyTorch, built around [semantic_comm_data_basics.ipynb](semantic_comm_data_basics.ipynb).

This project demonstrates how sentence meaning can be transmitted through a noisy channel by learning robust latent representations, then reconstructed at the receiver side.

---

## Table of Contents

- [Overview](#overview)
- [Core Idea](#core-idea)
- [Project Structure](#project-structure)
- [Pipeline Flow](#pipeline-flow)
- [Main Components](#main-components)
- [Training Strategy](#training-strategy)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [How to Run](#how-to-run)
- [Expected Behavior](#expected-behavior)
- [Important Terms](#important-terms)
- [Troubleshooting](#troubleshooting)
- [Limitations and Scope](#limitations-and-scope)
- [Future Improvements](#future-improvements)

---

## Overview

Conventional communication optimizes symbol/bit accuracy. This project targets **semantic fidelity**: preserving the intended meaning under channel noise.

The notebook includes:

1. Local Europarl-style sentence loading
2. Tokenization and vocabulary building
3. Dataset and masked batching
4. Baseline semantic communication model (`MiniSemanticComm`)
5. Improved encoder-decoder model (`Seq2SeqSemanticComm`)
6. Training/validation loops
7. Multi-SNR evaluation with language-quality metrics (BLEU, WER)

---

## Core Idea

The communication chain is modeled as:

```text
Text -> Tokens -> Semantic Encoder -> AWGN Channel -> Decoder -> Reconstructed Text
```

Noise is injected into **continuous latent representations** instead of raw tokens, allowing the model to learn channel-robust semantic features.

---

## Project Structure

```text
BTP_Thesis work/
├── semantic_comm_data_basics.ipynb      # Main traditional semantic communication notebook
├── Traditional_semantic_communication.md # Detailed write-up (companion doc)
├── europarl/                            # Local corpus root
│   └── en/en/*.txt                      # Europarl-style sentence files
├── .venv/                               # Local Python environment (optional)
└── hf_cache/                            # Local cache directory present in workspace
```

> This README intentionally documents only the traditional pipeline notebook.

---

## Pipeline Flow

### 1) Environment Setup
- Imports PyTorch, dataloader utilities, BLEU/WER helpers.
- Selects device (`cuda` if available else `cpu`).

### 2) Local Corpus Loading
- Reads files from `europarl/en/en`.
- Removes empty lines and XML-like tags.
- Builds sentence list for training and evaluation.

### 3) Text Preprocessing
- Lowercasing
- Regex cleanup (alphanumeric + whitespace)
- Whitespace tokenization

### 4) Vocabulary Construction
- Frequency-based token inclusion
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- `encode()` and `decode()` mapping utilities

### 5) Dataset + Batching
- Converts each sentence into teacher-forcing pair:
  - source: `ids[:-1]`
  - target: `ids[1:]`
- Pads variable-length sequences
- Creates `src_mask` (`1` real token, `0` pad)

### 6) Baseline Semantic Model
- Embedding + Transformer encoder
- AWGN channel in latent space
- Receiver projection + classifier
- Cross-entropy loss on target tokens

### 7) Upgraded Training
- Longer schedule with early stopping
- Best validation checkpoint restore

### 8) SNR-wise Evaluation
- Evaluates multiple channel qualities (e.g., 0/5/10/15 dB)
- Reports token accuracy, BLEU, WER, and validation loss

### 9) Stronger Seq2Seq Model
- Explicit Transformer encoder + decoder
- Teacher forcing during training
- Greedy decoding for inference demo

---

## Main Components

- `clean_and_tokenize(text)`
  - Normalizes and tokenizes raw text.

- `WordVocab`
  - Builds vocabulary and handles ID ↔ token conversion.

- `EuroparlTextDataset`
  - Prepares source/target token sequences.

- `make_collate_fn(pad_id)`
  - Pads batches and builds source mask.

- `AWGNChannel`
  - Adds Gaussian noise according to SNR.

- `MiniSemanticComm`
  - Lightweight baseline semantic model.

- `Seq2SeqSemanticComm`
  - Stronger encoder-decoder architecture with causal decoding.

---

## Training Strategy

### Baseline Stage
- Quick initial training loop for sanity and convergence checks.
- Reports `train_loss` and `val_loss` per epoch.

### Upgraded Stage
- Extended epochs and validation tracking.
- Early stopping based on validation loss patience.
- Restores best observed checkpoint.

### Seq2Seq Stage
- Teacher forcing with decoder input shifted by one token.
- Gradient clipping for stability.

---

## Evaluation and Metrics

### Cross-Entropy (CE) Loss
- Token-level objective.
- Lower is better.
- PAD tokens ignored using `ignore_index=PAD_ID`.

### Token Accuracy
- Fraction of correctly predicted non-pad tokens.
- Good for local correctness, weaker for global semantics.

### BLEU
- N-gram overlap between reference and prediction.
- Higher is better.
- Captures lexical/phrase similarity.

### WER (Word Error Rate)

$$
WER = \frac{S + D + I}{N}
$$

Where:
- $S$: substitutions
- $D$: deletions
- $I$: insertions
- $N$: number of words in reference

Lower is better, and `0.0` indicates exact word-level match.

---

## How to Run

1. Open [semantic_comm_data_basics.ipynb](semantic_comm_data_basics.ipynb).
2. Ensure dataset files exist under `europarl/en/en`.
3. Run all cells top-to-bottom.
4. Review:
   - baseline outputs
   - upgraded training logs
   - SNR evaluation table
   - one-sentence practical demo
   - stronger seq2seq demo

---

## Expected Behavior

As SNR increases:

- validation loss generally decreases
- token accuracy generally increases
- BLEU generally increases
- WER generally decreases

This indicates better semantic recovery over cleaner channels.

---

## Important Terms

- **Semantic Communication**: transmission focused on meaning preservation.
- **AWGN**: Additive White Gaussian Noise.
- **SNR (dB)**: channel quality indicator; higher means less noise impact.
- **Teacher Forcing**: decoder receives ground-truth prior token during training.
- **Greedy Decoding**: choose max-probability token at each step.
- **BOS/EOS/PAD/UNK**: sequence boundary and vocabulary control tokens.

---

## Troubleshooting

- **No local data found**
  - Verify `europarl/en/en/*.txt` exists.

- **Training errors after interruption**
  - Re-run model definition and loader cells before training cells.

- **Unstable loss**
  - Reduce learning rate, keep gradient clipping, and verify mask/pad handling.

- **Slow runtime**
  - Use GPU if available; reduce epochs and `MAX_EVAL_BATCHES` for quick checks.

---

## Limitations and Scope

- Word-level tokenization is simple and can miss richer linguistic structure.
- Evaluation emphasizes BLEU/WER and token-level behavior; deeper semantic scoring is limited.
- Domain transfer quality depends on training corpus characteristics.

---

## Future Improvements

- Subword tokenization (BPE/SentencePiece)
- Beam search decoding for stronger sequence quality
- Additional semantic similarity metrics
- More rigorous robustness testing across broader SNR ranges
- Better checkpoint/reporting utilities

---

If you are using this repository for thesis/demo purposes, this notebook is the full traditional baseline for semantic communication and can be extended into more advanced channel-robust architectures.