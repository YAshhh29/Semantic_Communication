# Traditional Semantic Communication (PyTorch)

This document is the **single consolidated project guide** for the traditional semantic communication implementation in `traditional_SC.ipynb`.

It combines a clean explanation-first structure with complete technical details for implementation, training, evaluation, and practical usage.

---

## Table of Contents

- [1) Project Objective](#1-project-objective)
- [2) Core Idea](#2-core-idea)
- [3) Workspace Structure](#3-workspace-structure)
- [4) End-to-End Function Flow](#4-end-to-end-function-flow)
- [5) Main Components and Functions](#5-main-components-and-functions)
- [6) Training Strategy](#6-training-strategy)
- [7) Metrics Used and Interpretation](#7-metrics-used-and-interpretation)
- [8) Expected Behavior Across SNR](#8-expected-behavior-across-snr)
- [9) How to Run](#9-how-to-run)
- [10) Important Terms (Glossary)](#10-important-terms-glossary)
- [11) Troubleshooting](#11-troubleshooting)
- [12) Limitations](#12-limitations)
- [13) Future Improvements](#13-future-improvements)
- [14) Scope](#14-scope)

---

## 1) Project Objective

Classical communication systems prioritize bit/symbol fidelity. This project focuses on **semantic fidelity**: preserving intended meaning when the channel is noisy.

The notebook demonstrates how text can be:

- converted into token representations,
- encoded into latent semantic features,
- perturbed through an AWGN channel,
- and reconstructed back into text,

while tracking quality with both token-level and sentence-level metrics.

---

## 2) Core Idea

Communication chain:

```text
Text -> Tokens -> Semantic Encoder -> AWGN Channel -> Decoder -> Reconstructed Text
```

Instead of corrupting raw words directly, noise is injected in **continuous latent space**. This allows the model to learn robust semantic representations that better survive channel degradation.

---

## 3) Workspace Structure

```text
BTP_Thesis work/
├── traditional_SC.ipynb                  # Main traditional semantic communication notebook
├── traditional_SC.md                     # This consolidated documentation
├── europarl/                             # Local Europarl-style corpus root
│   └── en/en/*.txt                       # Text files used for training/evaluation
├── .venv/                                # Local Python environment (optional)
└── hf_cache/                             # Local cache directory present in workspace
```

---

## 4) End-to-End Function Flow

The notebook follows this sequence:

1. **Environment + Imports**
   - Imports PyTorch, dataset utilities, and BLEU/WER tools.
   - Selects `cuda` or `cpu` runtime.

2. **Data Loading (Local Europarl)**
   - Reads `.txt` files from `europarl/en/en`.
   - Removes XML-like markup and empty lines.
   - Builds sentence list.

3. **Text Processing + Vocabulary**
   - `clean_and_tokenize(text)`: lowercase + regex cleanup + split.
   - `WordVocab`: frequency-filtered vocabulary.
   - Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.

4. **Dataset + DataLoader**
   - `EuroparlTextDataset`: creates teacher-forcing pairs.
   - `src = ids[:-1]`, `tgt = ids[1:]`.
   - Collate pads sequences and creates `src_mask`.

5. **Baseline Model (`MiniSemanticComm`)**
   - Token embedding
   - Transformer encoder
   - AWGN channel in latent space
   - Receiver projection + classifier

6. **Baseline Training + Validation**
   - Cross-entropy with PAD ignored (`ignore_index=PAD_ID`).
   - Gradient clipping.
   - Per-epoch train/val loss reporting.

7. **Upgraded Training Loop**
   - Longer training schedule.
   - Early stopping by validation patience.
   - Best-checkpoint restore.

8. **SNR-wise Evaluation**
   - Runs at multiple SNR values (e.g., `0, 5, 10, 15 dB`).
   - Reports loss, token accuracy, BLEU, WER.

9. **Single-Sentence Practical Demo**
   - Shows transmitter vs receiver behavior for one sentence across SNR values.

10. **Stronger Seq2Seq Model (`Seq2SeqSemanticComm`)**
    - Transformer encoder-decoder architecture.
    - Positional encoding + AWGN channel.
    - Teacher forcing in training, greedy decoding in inference.

---

## 5) Main Components and Functions

- `load_sentences_from_local(...)`
  - Reads and filters local corpus lines.

- `clean_and_tokenize(...)`
  - Text normalization and token splitting.

- `WordVocab`
  - Token-ID mapping and decode support.

- `EuroparlTextDataset`
  - Sequence-to-sequence sample construction.

- `make_collate_fn(pad_id)`
  - Batch padding and source mask generation.

- `AWGNChannel.forward(x, snr_db)`
  - Adds Gaussian noise according to SNR.

- `MiniSemanticComm`
  - Baseline semantic channel model.

- `Seq2SeqSemanticComm`
  - Stronger encoder-decoder semantic model.

---

## 6) Training Strategy

### Baseline Stage
- Quick loop to validate setup and convergence direction.
- Tracks `train_loss` and `val_loss`.

### Upgraded Stage
- Extended epochs, checkpointing, and patience-based early stopping.
- Restores best validation state for stable evaluation.

### Seq2Seq Stage
- Teacher forcing with shifted decoder inputs.
- Gradient clipping for optimization stability.

---

## 7) Metrics Used and Interpretation

### 7.1 Cross-Entropy Loss
- Token-level objective over vocabulary logits.
- Lower is better.
- PAD tokens excluded from penalty.

### 7.2 Token Accuracy
- Fraction of correctly predicted non-pad tokens.
- Useful for local correctness but not full semantic equivalence.

### 7.3 BLEU
- N-gram overlap between reference and reconstructed text.
- Higher is better.
- Reflects lexical and phrase-level similarity.

### 7.4 WER (Word Error Rate)

$$
WER = \frac{S + D + I}{N}
$$

Where:
- $S$: substitutions
- $D$: deletions
- $I$: insertions
- $N$: number of words in reference

Lower is better, and `0.0` indicates exact match.

---

## 8) Expected Behavior Across SNR

As SNR increases, typical trends are:

- validation loss decreases,
- token accuracy increases,
- BLEU increases,
- WER decreases.

This indicates better semantic recovery under cleaner channel conditions.

---

## 9) How to Run

1. Open `traditional_SC.ipynb`.
2. Ensure local text files are available at `europarl/en/en`.
3. Run cells from top to bottom.
4. Inspect:
   - baseline model outputs,
   - upgraded training logs,
   - SNR evaluation table,
   - one-sentence practical demo,
   - seq2seq demo outputs.

For quick iteration, reduce epochs and evaluation batch limits.

---

## 10) Important Terms (Glossary)

- **Semantic Communication**: communication focused on preserving meaning.
- **AWGN**: additive white Gaussian noise channel model.
- **SNR (dB)**: signal-to-noise ratio; higher values imply cleaner channel.
- **Teacher Forcing**: decoder receives ground-truth previous token during training.
- **Greedy Decoding**: picks highest-probability token at each step.
- **BOS / EOS / PAD / UNK**:
  - BOS: begin sequence
  - EOS: end sequence
  - PAD: padding token
  - UNK: unknown token

---

## 11) Troubleshooting

- **No corpus loaded**
  - Check `europarl/en/en/*.txt` availability.

- **Training failed after interruption**
  - Re-run model and dataloader definition cells before training cells.

- **Unstable loss**
  - Lower learning rate and verify masking/padding logic.

- **Slow execution**
  - Use GPU if available; reduce epochs and `MAX_EVAL_BATCHES`.

---

## 12) Limitations

- Word-level tokenization is simple and can miss richer subword semantics.
- BLEU/WER do not cover all dimensions of meaning preservation.
- Domain transfer quality depends on corpus distribution.

---

## 13) Future Improvements

- Subword tokenization (BPE/SentencePiece)
- Beam search decoding
- Additional semantic similarity metrics
- Broader robustness experiments over more channel conditions
- More structured checkpoint/version reporting

---

## 14) Scope

This document covers only the traditional semantic communication pipeline in:

- `traditional_SC.ipynb`

and intentionally excludes any separate LLM-backed notebook.
