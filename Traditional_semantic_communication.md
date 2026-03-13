# Traditional Semantic Communication (PyTorch) — Project README

This project implements a **traditional semantic communication pipeline** for text using PyTorch, centered on the notebook `semantic_comm_data_basics.ipynb`.

It demonstrates the full path from raw sentence data to noisy-channel transmission and reconstruction:

- local dataset loading
- text preprocessing and vocabulary building
- tensorized batching with masks
- encoder/channel/decoder modeling
- training and validation
- evaluation across multiple SNR points with semantic-quality metrics

---

## 1) Project Objective

Classical communication focuses on bit-level fidelity. This project explores **semantic communication**, where the objective is to preserve **meaning** under channel noise.

In this notebook, text is represented as tokens, encoded into latent representations, corrupted by an AWGN channel, and reconstructed back to text. Performance is measured with both token-level and sentence-level metrics.

---

## 2) Workspace Structure

Current folder structure:

```text
BTP_Thesis work/
├── semantic_comm_data_basics.ipynb      # Main traditional semantic communication notebook
├── europarl/                            # Local text corpus directory (Europarl-style)
├── .venv/                               # Local Python environment (if used)
└── hf_cache/                            # Cache directory present in workspace
```

> For this README, only `semantic_comm_data_basics.ipynb` is documented.

---

## 3) End-to-End Function Flow

The notebook follows this execution flow:

1. **Environment + Imports**
   - Imports PyTorch, utilities, and metrics dependencies.
   - Selects device (`cuda` if available, else `cpu`).

2. **Data Loading (Local Europarl)**
   - Reads `.txt` files from `europarl/en/en`.
   - Removes XML-like markup lines.
   - Keeps cleaned sentence lines.

3. **Text Processing + Vocabulary**
   - `clean_and_tokenize(text)`: lowercase + remove non-alphanumeric + split.
   - `WordVocab`: builds frequency-filtered vocabulary with special tokens:
     - `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
   - Provides `encode()` and `decode()` methods.

4. **Dataset + DataLoader**
   - `EuroparlTextDataset` converts sentence IDs into teacher-forcing pairs:
     - `src = ids[:-1]`
     - `tgt = ids[1:]`
   - Collate function pads variable-length sequences and builds `src_mask`.

5. **Mini Baseline Model**
   - `MiniSemanticComm` architecture:
     - token embedding
     - transformer encoder
     - AWGN latent channel
     - receiver projection + classifier
   - Forward pass outputs token logits.

6. **Baseline Training + Validation**
   - Cross-entropy loss with `ignore_index=PAD_ID`.
   - Gradient clipping for stability.
   - Reports per-epoch train/validation loss.

7. **Upgraded Training Loop**
   - Adds early stopping (patience) and best-checkpoint restore.
   - Tracks training history.

8. **SNR-wise Evaluation**
   - Evaluates at multiple SNR points (e.g., 0, 5, 10, 15 dB).
   - Reports:
     - validation loss
     - token accuracy
     - BLEU
     - WER

9. **Single-Sentence Practical Demo**
   - Shows reconstruction quality per SNR for one sentence.

10. **Stronger Seq2Seq Semantic Model**
    - Defines explicit encoder-channel-decoder architecture (`Seq2SeqSemanticComm`).
    - Uses positional encoding + transformer encoder/decoder + AWGN channel.
    - Trains with teacher forcing and restores best validation checkpoint.

11. **Seq2Seq SNR Demo**
    - Greedy decoding over channel outputs.
    - Sentence-level BLEU/WER across SNR values.

---

## 4) Model Working (Conceptual)

### 4.1 Semantic Transmission Loop

```text
Input sentence
   -> tokenization + IDs
   -> semantic encoder
   -> latent representation
   -> AWGN channel noise (SNR-controlled)
   -> receiver/decoder
   -> reconstructed token sequence
   -> reconstructed sentence
```

### 4.2 Why AWGN in Latent Space?

Instead of corrupting raw text directly, this setup perturbs **continuous semantic vectors**, simulating channel impairment while keeping the NLP pipeline differentiable for learning robust representations.

---

## 5) Important Components and Functions

- `load_sentences_from_local(...)`  
  Loads and filters local corpus text.

- `clean_and_tokenize(...)`  
  Normalizes input text for vocabulary consistency.

- `WordVocab`  
  Handles token-to-id and id-to-token mappings with special tokens.

- `EuroparlTextDataset`  
  Builds sequence pairs for next-token prediction style training.

- `make_collate_fn(pad_id)`  
  Pads variable-length sequences and creates attention masks.

- `AWGNChannel.forward(x, snr_db)`  
  Adds Gaussian noise based on requested SNR.

- `MiniSemanticComm`  
  Lightweight semantic communication baseline.

- `Seq2SeqSemanticComm`  
  Stronger encoder-decoder model with autoregressive decoding support.

---

## 6) Key Terms (Quick Glossary)

- **Semantic Communication**: Communication aimed at preserving meaning rather than exact symbol sequence.
- **AWGN (Additive White Gaussian Noise)**: Gaussian noise model added to signal/latent vectors.
- **SNR (Signal-to-Noise Ratio, dB)**: Higher SNR means cleaner channel.
- **Teacher Forcing**: During training, decoder receives ground-truth previous token.
- **BOS / EOS / PAD / UNK**:
  - BOS = beginning of sentence
  - EOS = end of sentence
  - PAD = padding for batching
  - UNK = out-of-vocabulary token
- **Greedy Decoding**: Select highest-probability token at each generation step.

---

## 7) Metrics Used and Interpretation

### 7.1 Cross-Entropy Loss
Training objective over vocabulary logits:

- Lower is better.
- Computed token-wise.
- PAD positions are ignored (`ignore_index=PAD_ID`).

### 7.2 Token Accuracy
Fraction of correctly predicted non-pad tokens.

- Useful for local token fidelity.
- Does not fully capture sentence-level meaning.

### 7.3 BLEU (Sentence-level average)
Measures n-gram overlap between reference and reconstruction.

- Higher is better.
- Sensitive to lexical overlap and phrasing.

### 7.4 WER (Word Error Rate)
Measures edit distance at word level:

\[
WER = \frac{S + D + I}{N}
\]

where:
- \(S\): substitutions
- \(D\): deletions
- \(I\): insertions
- \(N\): words in reference

- Lower is better.
- 0 means perfect match.

---

## 8) Training and Evaluation Behavior

Typical expected trends as SNR increases:

- validation loss decreases
- token accuracy increases
- BLEU increases
- WER decreases

This reflects better semantic preservation under cleaner channel conditions.

---

## 9) How to Run

1. Open `semantic_comm_data_basics.ipynb`.
2. Run cells top-to-bottom in order.
3. Ensure local Europarl text files exist under `europarl/en/en`.
4. Use GPU if available for faster training.
5. For quick checks, reduce:
   - epochs
   - max eval batches
   - model size (if needed)

---

## 10) Practical Notes

- The notebook contains both a **minimal baseline** and a **stronger seq2seq variant**.
- Some training cells may error if interrupted or if runtime state is stale; rerun preceding model/data definition cells before retrying.
- Results are stochastic; set random seeds for reproducibility as already done in the notebook.

---

## 11) Scope of This README

This README documents only the traditional semantic communication pipeline in:

- `semantic_comm_data_basics.ipynb`

and intentionally excludes all discussion of any separate LLM-backed notebook.
