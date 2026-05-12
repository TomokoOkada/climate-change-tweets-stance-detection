# Climate Change Stance Detection Pipeline

A multi-stage NLP pipeline that classifies climate change discourse on Twitter
across four stance dimensions. Built to support a computational social science
study on how rural–urban context shapes climate change attitudes in the U.S.

**Tech stack:** Python · HuggingFace Transformers · RoBERTa · Kaggle T4 GPU  
**Scale:** ~1.16 million tweets processed  
**Training data:** 4,000 LLaMA 3-annotated tweets, human-validated

---

## What This Project Does

### The Research Problem

Most tools that analyze social media either do simple keyword counting ("does
this tweet mention 'climate change'?") or apply off-the-shelf sentiment
analysis. Neither captures what researchers actually need: *does this tweet
take a specific stance, and on which aspect of the debate?*

This pipeline implements a **codebook-driven classification scheme** developed
for systematic annotation of climate tweets. Four questions are answered for each
tweet:

| # | Question | Why It Matters |
|---|----------|---------------|
| **Q1** | Is this tweet actually about climate change? | Removes noise from ~1.16M tweets |
| **Q2** | Does it say whether climate change is real? | Captures denial vs. acceptance of CC existence |
| **Q3** | Does it say whether humans cause it? | Captures denial vs. acceptance of human causality |
| **Q4** | Does it say whether humans should act? | Captures calls for (or against) climate action |

For Q2–Q4, a tweet only gets classified if it makes an **explicit** stance
statement — just mentioning climate change doesn't count. This required a
two-stage approach for each question (see below).

**Annotation methodology:** 4,000 tweets were annotated using LLaMA 3 (May 2024)
following an iterative validation protocol. The researcher annotated batches of
100 tweets independently, then compared against LLaMA 3 outputs, refined the
prompt, and repeated. After 6–7 cycles, LLM–human agreement reached ~90%, at
which point LLaMA 3 was used to annotate the full training set. Inter-annotator
agreement was assessed as percent agreement across validation batches (formal
Cohen's kappa was not computed).

---

## Pipeline Architecture

```
~1.16M tweets
      │
      ▼
┌──────────────────────────────────────────┐
│  Q1: Is this tweet about climate change? │  → 83.88% accuracy
│  3 categories: Not related / Partial /   │
│  Primarily about climate change          │
└────────────────┬─────────────────────────┘
                 │  ~360k climate-relevant tweets pass
                 ▼
    ┌────────────┬────────────┐
    ▼            ▼            ▼
  Q2           Q3           Q4
  ──────────────────────────────────────────
  Stage 1: Does this tweet explicitly address the question?
  (Binary: "Mentioned" vs "Not Mentioned")
  │
  └─ Only "Mentioned" tweets go to Stage 2
  │
  Stage 2: What stance does it take?
  (3-class: Denial / Acceptance / Indeterminate)
  ──────────────────────────────────────────
```

### Why Two Stages?

The first version of this pipeline tried to classify all tweets into four
categories at once (Denial / Acceptance / Indeterminate / Not Mentioned).
It didn't work well — the "Not Mentioned" class was 51–76% of tweets depending
on the question, and the model learned to predict it for almost everything.

Splitting into **Stage 1** (binary: does this tweet even address the question?)
and **Stage 2** (what does it say?) solved the problem. The binary filter is
trained to be aggressive — it's better to pass a borderline tweet to Stage 2
(where it'll be correctly classified) than to drop it entirely.

---

## Results

| Script | Task | Accuracy | Key Finding |
|--------|------|----------|-------------|
| `q1_fine_tuning.py` | Climate relevance filter | **83.88%** | 5 epochs optimal; 7+ caused overfitting |
| `q2_step1_fine_tuning.py` | Q2 mention detection | 70.75% overall / **90.51% recall** | Low accuracy is intentional — optimized for recall |
| `q2_step2_fine_tuning.py` | Q2 stance (3-class) | **83.85%** | 7 approaches tried; roberta-large beat BERTweet by 2.8% |
| `q3_step1_fine_tuning.py` | Q3 mention detection | **82.88%** | Oversampling beat class weighting by 0.4% |
| `q3_step2_fine_tuning.py` | Q3 stance (3-class) | **82.98%** | Two-stage transfer learning: +7.45% over single-stage |
| `q4_step1_fine_tuning.py` | Q4 mention detection | **80.38%** | 4 notebook versions; oversampling attempt gave 29.62% — double-correction bug (oversampling + class weights both applied), fully explained in script |
| `q4_step2_fine_tuning.py` | Q4 stance (3-class) | **81.86%** | Mean across 5 seeds: 77.64% ± 3.19%; ensemble (80.17%) lower than best seed — variance disclosed; see script for paper-reporting guidance |

---

## Files in This Repository

All scripts follow the same pattern: load data → clean → split → tokenize →
train → evaluate → save model.

```
q1_fine_tuning.py        Q1 relevance classifier (training)
q1_inference.py          Q1 relevance classifier (run on new data)

q2_step1_fine_tuning.py  Q2 Stage 1: mention detection (training)
q2_step1_inference.py    Q2 Stage 1: mention detection (run on new data)
q2_step2_fine_tuning.py  Q2 Stage 2: stance classification (training)
q2_step2_inference.py    Q2 Stage 2: stance classification (run on new data)

q3_step1_fine_tuning.py  Q3 Stage 1: mention detection (training)
q3_step1_inference.py    Q3 Stage 1: mention detection (run on new data)
q3_step2_fine_tuning.py  Q3 Stage 2: stance classification (training)
q3_step2_inference.py    Q3 Stage 2: stance classification (run on new data)

q4_step1_fine_tuning.py  Q4 Stage 1: mention detection (training)
q4_step1_inference.py    Q4 Stage 1: mention detection (run on new data)
q4_step2_fine_tuning.py  Q4 Stage 2: stance classification (training)
q4_step2_inference.py    Q4 Stage 2: stance classification (run on new data)
```

---

## What Was Tried to Improve Each Model

Each model went through its own experimentation process. The full trial-and-error
log is in each script's docstring. Here is the summary.

---

### Q1 — Climate Relevance (`q1_fine_tuning.py`)

**Model:** `cardiffnlp/twitter-roberta-base`  
Twitter-RoBERTa was pre-trained on ~58M tweets, making it a natural fit. It
worked well from the start, so most experimentation was on **epoch count**:

| Epochs | Result |
|--------|--------|
| 3 | Underfitting (~79%) |
| **5** | **83.88% ← used this** |
| 7 | Val loss starts rising while train loss drops (overfitting) |
| 10, 15, 20 | Worse |

No class weighting needed — the Q1 label distribution was reasonably balanced.

---

### Q2 Stage 1 — Mention Detection (`q2_step1_fine_tuning.py`)

**Model:** `roberta-base`  
Originally planned to use `twitter-roberta-base`, but it threw persistent
tokenizer errors in Kaggle's environment. Switched to `roberta-base` with
minimal performance impact.

**Key design decision:** this model was optimized for *recall on the Mentioned
class*, not overall accuracy. The logic: a false positive (Not Mentioned
predicted as Mentioned) goes to Stage 2 where it gets corrected. A false
negative (Mentioned predicted as Not Mentioned) is lost permanently.

Result: 70.75% overall accuracy, but 90.51% recall on the Mentioned class.
The 70.75% looks bad at first glance — it's intentional.

---

### Q2 Stage 2 — Stance Classification (`q2_step2_fine_tuning.py`)

**Final model:** `roberta-large`, 10 epochs — **83.85%**

Seven approaches were tried in order:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| BERTweet baseline | 81.03% | Good start, but minority classes underperformed |
| BERTweet + class weighting | 75.13% | **Worse by 5.9%** — too aggressive |
| BERTweet + more epochs, lower LR | 79.23% | Didn't recover |
| **roberta-large** | **83.85%** | ← Final choice |
| roberta-large + 5 more epochs | 82.56% | Model had already converged |
| Twitter-sentiment transfer | 78.97% | Sentiment ≠ stance |
| Hyperparameter search (16 configs) | 82.05% best | No improvement over run 2 |

**Takeaway:** Model capacity (roberta-large: 355M params vs BERTweet's 135M)
mattered more than Twitter-specific pre-training for nuanced 3-class stance
classification.

---

### Q3 Stage 1 — Mention Detection (`q3_step1_fine_tuning.py`)

**Final model:** `roberta-large` + random oversampling — **82.88%**

Q3 has a harder imbalance than Q2 Stage 1 (76% Not Mentioned vs 51%).
Three approaches were compared directly:

| Approach | Accuracy |
|----------|----------|
| BERTweet | 81.12% |
| roberta-large + class weighting | 82.50% |
| **roberta-large + random oversampling (50/50 balance)** | **82.88%** |

Oversampling duplicated the minority class (Mentioned tweets) until the
training set was balanced. The extra 0.38% over class weighting was small but
consistent.

---

### Q3 Stage 2 — Stance Classification (`q3_step2_fine_tuning.py`)

**Final model:** Two-stage transfer learning (Q2→Q3) — **82.98%**

This was the most challenging model. After filtering to "Mentioned" tweets,
only **938 training samples** remained — far too few for roberta-large to
converge reliably. Single-stage training gave 75.53%.

Five approaches were tried:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| BERTweet | 72.34% | Small dataset punished smaller model |
| roberta-large single-stage | 75.53% | Val loss climbing — overfitting |
| roberta-large + oversampling | 74.47% | **Worse** — duplicating 89 samples 6x caused severe overfit |
| roberta-large + class weighting | 75.00% | Marginal improvement |
| **Two-stage: pre-train on Q2, fine-tune on Q3** | **82.98%** | +7.45% |

**Why use Q2 data for pre-training Q3?**
- Q2 and Q3 have the same 3-class label structure (Denial/Acceptance/Indeterminate)
- Both are about the *scientific basis* of climate change — Q2 asks "is it
  real?", Q3 asks "did humans cause it?" — the vocabulary and framing overlap
- Q2 has 1,949 "Mentioned" samples, roughly 2× Q3's 938

Pre-training on Q2 gave the model a much better initialization than random
weights before adapting to the smaller Q3 dataset.

---

### Q4 Stage 1 — Mention Detection (`q4_step1_fine_tuning.py`)

**Final model:** `roberta-large` + WeightedTrainer + two-stage (Q3→Q4) — **80.38%**

This took 4 notebook versions to get right. Attempts in order:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| BERTweet | 74.75% | Starting point |
| Two-stage Q3→Q4, no weighting | 79.38% | Good jump |
| Two-stage + class weighting, 10 epochs | 76.75% | **Worse** — weights disrupted Stage 1 patterns |
| Two-stage + oversampling | **29.62%** | Double-correction bug: oversampling balanced training to 50/50, but class weights computed on original 70/30 remained active — model predicted "Mentioned" for nearly everything; 29.62% ≈ Mentioned base rate on unbalanced val set. Fully explained in script. |
| 3-model ensemble | 78.88% | High seed variance; averaging hurt the best model |
| Seed stability test (5 seeds) | 70–80% | Only seed=42 reliably approached 80% |
| **Two-stage + WeightedTrainer, 8 epochs** | **80.38%** | ← Final |

**Key insight:** reducing Stage 2 from 10 to 8 epochs was crucial — validation
loss diverged clearly after epoch 8. `load_best_model_at_end=True` recovered
the best checkpoint automatically.

**Why Q3 for pre-training Q4 (not Q2)?**
- Q3 and Q4 both have similar class imbalance (~76% and ~70% Not Mentioned)
- Q2 is nearly balanced (51%) — pre-training on it would give misleading priors
- Q3 ("human causality") and Q4 ("human responsibility to act") are both about
  human agency; Q2 ("is CC real?") is conceptually further away

**What is WeightedTrainer?**  
A custom training class that applies penalty weights to the loss function
inversely proportional to class frequency. With 70% "Not Mentioned" in the
training data, an unweighted model learns that predicting "Not Mentioned"
for everything gives 70% accuracy — and stops there. WeightedTrainer forces
the model to take errors on the rare "Mentioned" class more seriously:
- Class 0 (Not Mentioned): weight 0.71× (penalized less)
- Class 1 (Mentioned): weight 1.69× (penalized more)

---

### Q4 Stage 2 — Stance Classification (`q4_step2_fine_tuning.py`)

**Final model:** `roberta-large` + WeightedTrainer, seed=42 — **81.86%**

This model had the most seed sensitivity in the entire pipeline:

| Seed | Accuracy |
|------|----------|
| **42** | **81.86%** |
| 123 | 77.64% |
| 456 | 77.22% |
| 789 | 78.48% |
| 2024 | 73.00% |

A 5-model ensemble averaged 80.17% — *lower* than seed=42 alone. Averaging
pulled the strong model down rather than lifting the weaker ones. The
documented ceiling for this dataset appears to be ~81–82%.

Other approaches tried:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| BERTweet | 77.22% | Same as Focal Loss |
| Focal Loss (γ=2.0) | 77.22% | No improvement over standard cross-entropy |
| Extended training (18 epochs) | 80.59% best | Val loss diverged around epoch 10 |
| Hyperparameter search (11 configs) | 81.43% best | Marginal improvement |
| **roberta-large + WeightedTrainer, seed=42** | **81.86%** | ← Final |

---

## Key Technical Decisions Across the Pipeline

### Why a different imbalance-handling technique for each model?

No single technique worked across all stages — the right choice depended on
dataset size and imbalance severity:

| Stage | Imbalance Ratio | Technique | Why |
|-------|-----------------|-----------|-----|
| Q2 Stage 1 | 1.05:1 (balanced) | None | No correction needed |
| Q2 Stage 2 | 6.1:1 | None | Class weighting *hurt* here (-5.9%) |
| Q3 Stage 1 | 3.25:1 | Oversampling | +0.4% over class weighting |
| Q3 Stage 2 | 4.3:1 | Transfer learning | Dataset too small; balancing made overfit worse |
| Q4 Stage 1 | 2.4:1 | WeightedTrainer | Part of the config that reached 80.38% |
| Q4 Stage 2 | 5.7:1 | WeightedTrainer | Needed for minority class performance |

---

## Setup

```bash
pip install -r requirements.txt
```

**Training data format:** A CSV with tweet text (column name: `title`) and
label columns `label_q1`, `label_q2`, `label_q3`, `label_q4`. Labels are
0/1/2 for stance categories, 9 for "Not Mentioned" (Q2–Q4 only).

Update `TRAINING_DATA_PATH` at the top of each fine-tuning script, then run:

```bash
python q1_fine_tuning.py
python q2_step1_fine_tuning.py
python q2_step2_fine_tuning.py
python q3_step1_fine_tuning.py
python q3_step2_fine_tuning.py   # needs label_q2 and label_q3 in same CSV
python q4_step1_fine_tuning.py   # needs label_q3 and label_q4 in same CSV
python q4_step2_fine_tuning.py
```

**GPU requirements:** roberta-large requires ~6–7GB VRAM. Developed on
Kaggle's T4 (15.9GB) with batch sizes of 8–16.

---

## Reproducibility Notes

- **Q4 Stage 2 is seed-sensitive.** seed=42 gave 81.86%; other seeds (123, 456, 789, 2024) gave 73–78%; **mean across 5 seeds: 77.64% ± 3.19%**. A 5-model ensemble gave 80.17% — lower than seed=42 alone, confirming the variance is real and not reducible by averaging. We report seed=42 with full disclosure of the distribution. Any downstream analysis should treat Q4 Stage 2 as operating at ~78% (mean), not 81.86%. See `q4_step2_fine_tuning.py` for paper-reporting language.
- **Q2 Stage 2 has ~0.8% run-to-run variance** (83.08% vs 83.85% on same config).
- All scripts set `seed=42` explicitly.

---

## License

MIT
