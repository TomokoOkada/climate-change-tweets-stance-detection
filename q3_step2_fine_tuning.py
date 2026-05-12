"""
Q3 Step 2 Fine-Tuning: Human Causality — Stance Classification
==============================================================
3-class classifier for tweets where a human-causality stance was detected
(Q3 Step 1 = "Mentioned").

Classification:
    0 (Denies):        Explicitly states climate change is NOT caused by humans
    1 (Affirms):       Explicitly states climate change IS caused by humans
    2 (Indeterminate): Both stances present, or stance is ambiguous

--------------------------------------------------------------------
FULL EXPERIMENT LOG
--------------------------------------------------------------------

Training data for Q3 Step 2 was small — only 938 labeled samples after
filtering to "Mentioned" tweets. Class distribution:
    0 (Denies):        156 / 938 (16.6%)
    1 (Affirms):       670 / 938 (71.4%)   <- dominant
    2 (Indeterminate): 112 / 938 (11.9%)
    Imbalance ratio:   6.02x

The small dataset size drove all the experimentation below.

1. BERTweet (vinai/bertweet-base, 10 epochs, LR=2e-5)
   -> 72.34% accuracy
   Poorest result. Small training set punished the smaller model.

2. RoBERTa-large, standard single-stage (10 epochs, LR=2e-5)
   -> 75.53% accuracy, F1=0.7411
   Better than BERTweet but val loss climbed steadily — overfitting
   on only 750 training samples.

3. RoBERTa-large + random oversampling (balanced 33.3% each class)
   Oversampled classes 0 and 2 to match class 1 (536 each -> 1,608 total).
   -> 74.47% accuracy (-1.06%)   <- WORSE
   Val loss jumped to 4.1 (severe overfitting). Duplicating 447 samples
   for class 2 (from only 89) meant the model saw the same tweets ~6x.
   The overfit was much worse than Q3 Step 1 because the dataset is
   half the size.

4. RoBERTa-large + class weighting
   Weights: Denies=2.0x, Affirms=0.47x, Indeterminate=2.81x
   -> 75.00% accuracy (-0.53%)   <- WORSE than baseline
   Neither oversampling nor class weighting helped. The problem
   was simply dataset size, not class imbalance per se.

5. Two-stage transfer learning (Q2 -> Q3)   <- FINAL MODEL
   Key insight: Q2 Step 2 classifies structurally identical stances
   (denial/acceptance/indeterminate) in similar tweet text.
   Q2 had 1,949 "Mentioned" samples — ~2x more than Q3's 938.

   Why Q2 for Q3 pre-training (and not Q4)?
   - Same label structure: Q2 and Q3 both use 3-class stance
     (Denial/Acceptance/Indeterminate). Q4 is also 3-class, but the
     conceptual distance is greater.
   - Conceptual adjacency: Q2 ("is CC real?") and Q3 ("caused by humans?")
     are both about the scientific basis of climate change. Skeptical tweets
     for Q2 and Q3 share vocabulary (hoax claims, denial framing); acceptance
     tweets share references to observed effects and scientific consensus. Q4
     (calls for action) is about policy response — conceptually further from
     Q3 than Q2 is.
   - Data size advantage: Q2's 1,949 "Mentioned" samples are ~2x Q3's 938,
     providing a stronger initialization for the classification head.

   Stage 1: Pre-train on Q2 Step 2 data (5 epochs, LR=2e-5)
            -> Model learns general stance classification patterns
   Stage 2: Fine-tune on Q3 Step 2 data (10 epochs, LR=1e-5)
            -> Model adapts to human-causality framing

   -> 82.98% accuracy, F1=0.8258   <- +7.45% improvement!

Why the two-stage approach worked so well here but wasn't needed for
Q2 or Q4: Q3's training set (938 samples) was simply too small for
roberta-large to converge reliably. Q2's 1,949 structurally similar
samples provided a much better initialization than random weights.
This is essentially domain-adaptive pre-training at a small scale.

--------------------------------------------------------------------
IMPLEMENTATION NOTE
--------------------------------------------------------------------

This script requires BOTH Q2 and Q3 labels in the master training CSV.
In our dataset, both were available in the same file.
If you only have Q3 labels, remove Stage 1 and train single-stage —
expect ~75.53% accuracy.

Author: [Your Name]
Date: November 2025
"""

import os
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TRAINING_DATA_PATH = "/kaggle/input/your-dataset/training_data.csv"
OUTPUT_MODEL_PATH  = "./q3_step2_fine_tuned_model"

# roberta-large: outperformed BERTweet (72.34%) across all tests.
# Two-stage training with this model achieved 82.98%.
BASE_MODEL = "roberta-large"
NUM_LABELS = 3
MAX_LENGTH = 128

# Stage 1: higher LR, no validation (just building stance representations)
STAGE1_EPOCHS = 5
STAGE1_LR     = 2e-5

# Stage 2: lower LR to gently adapt without forgetting Stage 1 patterns
STAGE2_EPOCHS = 10
STAGE2_LR     = 1e-5

BATCH_SIZE   = 16
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
RANDOM_SEED  = 42

os.environ["WANDB_DISABLED"] = "true"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_training_data(filepath: str) -> pd.DataFrame:
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"✓ Loaded with {enc} encoding — {len(df):,} rows")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not load {filepath}")


def prepare_q2_stage1_data(df: pd.DataFrame, text_col: str) -> tuple:
    """
    Extract Q2 Step 2 data for Stage 1 pre-training.

    Q2 labels 0/1/2 map to the same stance categories as Q3:
        0: Denial/Denies      -> learn denial stance patterns
        1: Acceptance/Affirms -> learn acceptance stance patterns
        2: Indeterminate      -> learn ambiguity patterns

    This structural similarity is why the transfer works.
    Q2 has 1,949 "Mentioned" samples — 2x more than Q3's 938.
    """
    df_q2 = df[df['label_q2'].isin([0, 1, 2])].copy()
    texts  = df_q2[text_col].fillna('').astype(str).tolist()
    labels = df_q2['label_q2'].astype(int).tolist()

    print(f"\nQ2 pre-training data ({len(texts):,} samples):")
    for lbl, name in {0: "Denial", 1: "Acceptance", 2: "Indeterminate"}.items():
        count = labels.count(lbl)
        print(f"  {lbl} ({name}): {count:,} ({count/len(labels)*100:.1f}%)")
    return texts, labels


def prepare_q3_stage2_data(df: pd.DataFrame, text_col: str) -> tuple:
    """
    Extract Q3 Step 2 data for Stage 2 fine-tuning.

    938 total samples, 6:1 imbalance (Affirms vs Indeterminate).
    We do NOT oversample or weight — both made things worse.
    The two-stage approach handles the small size problem instead.
    """
    df_q3 = df[df['label_q3'].isin([0, 1, 2])].copy()
    texts  = df_q3[text_col].fillna('').astype(str).tolist()
    labels = df_q3['label_q3'].astype(int).tolist()

    print(f"\nQ3 fine-tuning data ({len(texts):,} samples):")
    for lbl, name in {0: "Denies", 1: "Affirms", 2: "Indeterminate"}.items():
        count = labels.count(lbl)
        print(f"  {lbl} ({name}): {count:,} ({count/len(labels)*100:.1f}%)")
    return texts, labels


# ==============================================================================
# TOKENIZATION & METRICS
# ==============================================================================

def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        texts, truncation=True, padding='max_length',
        max_length=max_length, return_tensors=None
    )
    dataset = Dataset.from_dict({
        'input_ids':      encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels':         labels
    })
    dataset.set_format('torch')
    return dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}


# ==============================================================================
# MAIN: TWO-STAGE TRAINING
# ==============================================================================

def train_q3_step2_model(
    training_data_path: str,
    output_path: str,
    text_column: str = 'title'
):
    print("=" * 70)
    print("Q3 STEP 2 FINE-TUNING: Human Causality — Two-Stage Training")
    print("=" * 70)
    print("Stage 1: Pre-train on Q2 data (1,949 samples, structurally similar)")
    print("Stage 2: Fine-tune on Q3 data (938 samples, target task)")
    print("Baseline single-stage: 75.53% | Two-stage result: 82.98% (+7.45%)")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading training data...")
    df = load_training_data(training_data_path)
    q2_texts, q2_labels = prepare_q2_stage1_data(df, text_column)
    q3_texts, q3_labels = prepare_q3_stage2_data(df, text_column)

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"\n[2/6] Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ── Tokenize Q2 (no train/val split — full dataset for pre-training) ──
    print("\n[3/6] Tokenizing Stage 1 data (Q2)...")
    q2_dataset = tokenize_data(q2_texts, q2_labels, tokenizer, MAX_LENGTH)

    # ── Tokenize Q3 (with validation split for Stage 2) ───────────────────
    print("\n[4/6] Tokenizing Stage 2 data (Q3)...")
    q3_train_texts, q3_val_texts, q3_train_labels, q3_val_labels = train_test_split(
        q3_texts, q3_labels,
        test_size=0.2, random_state=RANDOM_SEED, stratify=q3_labels
    )
    print(f"  Q3 train: {len(q3_train_texts):,}  |  Q3 val: {len(q3_val_texts):,}")
    q3_train_dataset = tokenize_data(q3_train_texts, q3_train_labels, tokenizer, MAX_LENGTH)
    q3_val_dataset   = tokenize_data(q3_val_texts,   q3_val_labels,   tokenizer, MAX_LENGTH)

    del df, q2_texts, q3_texts, q3_train_texts, q3_val_texts
    gc.collect()

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[5/6] Two-stage training...")
    print(f"\n  Stage 1: Pre-training on Q2 data...")
    print(f"  {len(q2_dataset):,} samples | {STAGE1_EPOCHS} epochs | LR={STAGE1_LR}")
    print(f"  Expected time: ~8-10 min on T4 GPU")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Stage 1: no evaluation, just building stance representations from Q2
    stage1_args = TrainingArguments(
        output_dir='./q3_step2_stage1_checkpoints',
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=STAGE1_LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        evaluation_strategy='no',
        save_strategy='epoch',
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to='none'
    )

    Trainer(
        model=model, args=stage1_args,
        train_dataset=q2_dataset,
        compute_metrics=compute_metrics
    ).train()

    print("✓ Stage 1 complete")

    # Stage 2: fine-tune on Q3 with lower LR
    print(f"\n  Stage 2: Fine-tuning on Q3 data...")
    print(f"  {len(q3_train_dataset):,} train samples | {STAGE2_EPOCHS} epochs | LR={STAGE2_LR}")
    print(f"  Lower LR preserves Stage 1 stance patterns")
    print(f"  Expected time: ~6-8 min on T4 GPU")

    stage2_args = TrainingArguments(
        output_dir='./q3_step2_stage2_checkpoints',
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=STAGE2_LR,
        warmup_steps=50,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        evaluation_strategy='epoch',
        save_strategy='no',
        fp16=torch.cuda.is_available(),
        report_to='none'
    )

    trainer_stage2 = Trainer(
        model=model, args=stage2_args,
        train_dataset=q3_train_dataset,
        eval_dataset=q3_val_dataset,
        compute_metrics=compute_metrics
    )
    trainer_stage2.train()

    # ── Evaluate & save ────────────────────────────────────────────────────
    print("\n[6/6] Final evaluation...")
    eval_results = trainer_stage2.evaluate()
    print(f"\nFinal accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Final F1:       {eval_results['eval_f1']:.4f}")
    print(f"\n(Two-stage baseline: 82.98% | Single-stage baseline: 75.53%)")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")
    return model, tokenizer


if __name__ == "__main__":
    # =========================================================================
    # THINGS TO UPDATE:
    # =========================================================================
    # 1. TRAINING_DATA_PATH — master CSV with BOTH label_q2 and label_q3
    # 2. text_column — tweet text column ('title' in our data)
    #
    # IF YOU ONLY HAVE Q3 LABELS (no Q2 data for Stage 1):
    # Skip Stage 1 and train directly on Q3. Expect ~75.53% accuracy.
    # Neither class weighting nor oversampling improved on this baseline.
    #
    # EXPECTED RESULTS WITH TWO-STAGE:
    # ~82-83% accuracy (we got 82.98%)
    # Training time: ~15-18 min total on T4 GPU
    # =========================================================================

    model, tokenizer = train_q3_step2_model(
        training_data_path=TRAINING_DATA_PATH,
        output_path=OUTPUT_MODEL_PATH,
        text_column='title'
    )
    print("\n" + "=" * 70)
    print("Q3 STEP 2 FINE-TUNING COMPLETE!")
    print("=" * 70)
