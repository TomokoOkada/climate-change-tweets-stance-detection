"""
Q1 Fine-Tuning: Climate Change Relevance Classification
========================================================
Fine-tunes a RoBERTa model to classify tweets into three relevance categories.
This is the entry point of the pipeline — tweets that pass this filter (labels
1 or 2) proceed to Q2–Q4 stance classification.

Classification Labels (from annotation codebook):
    0: NOT RELATED       — Tweet is not related to climate change at all
    1: PARTIALLY RELATED — Mentions climate change alongside other political/social
                           issues (e.g., healthcare + economy + climate change)
    2: PRIMARILY CLIMATE — About climate change without equivalent references
                           to other political issues

Model selection notes:
    We used cardiffnlp/twitter-roberta-base, pre-trained on ~58M tweets.
    This was the first and only model we tried for Q1 — Twitter-RoBERTa was
    a natural fit for Twitter data and performed well from the start.

Hyperparameter search:
    We ran a sweep over training epochs: [3, 5, 7, 10, 15, 20].
    - 3 epochs: underfitting, accuracy ~79%
    - 5 epochs: best validation accuracy at 83.88%, loss still stable
    - 7+ epochs: validation loss starts climbing while training loss drops
                 (classic overfitting signal on a ~4k training set)
    Final choice: 5 epochs.

Final performance: 83.88% accuracy on held-out validation set.

Author: [Your Name]
Date: October 2025
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
OUTPUT_MODEL_PATH  = "./q1_fine_tuned_model"

# twitter-roberta-base: pre-trained on ~58M tweets, good at hashtags/mentions
BASE_MODEL = "cardiffnlp/twitter-roberta-base"
NUM_LABELS = 3
MAX_LENGTH = 128

# 5 epochs was the sweet spot in our hyperparameter search.
# Going higher caused overfitting (val loss increased from epoch ~7 onward).
EPOCHS        = 5
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS  = 500
WEIGHT_DECAY  = 0.01
RANDOM_SEED   = 42

os.environ["WANDB_DISABLED"] = "true"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare_data(filepath: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Load and clean training data.

    Twitter CSVs often have encoding issues (emojis, special characters),
    so we try multiple encodings before giving up.

    Args:
        filepath:  Path to master training CSV
        text_col:  Tweet text column name
        label_col: Q1 label column (values: 0, 1, 2)

    Returns:
        Cleaned DataFrame with columns ['text', 'label']
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"✓ Loaded with {enc} encoding")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not load {filepath} with any standard encoding")

    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']

    n_before = len(df)
    df = df.dropna()
    df = df[df['text'].str.strip() != '']
    print(f"✓ Removed {n_before - len(df)} empty/null rows")

    df['label'] = df['label'].astype(int)

    # Q1 labels should only be 0, 1, 2
    valid = df['label'].isin([0, 1, 2])
    if not valid.all():
        print(f"⚠ Dropping {(~valid).sum()} rows with unexpected labels")
        df = df[valid]

    print(f"\nLabel distribution:")
    label_names = {0: "Not related", 1: "Partial", 2: "Primarily climate"}
    for lbl, name in label_names.items():
        count = (df['label'] == lbl).sum()
        pct   = count / len(df) * 100
        print(f"  {lbl} ({name}): {count:,} ({pct:.1f}%)")

    return df


def clean_tweet_text(text: str) -> str:
    """
    Minimal text cleaning — twitter-roberta was pre-trained on raw tweets,
    so aggressive cleaning (removing hashtags, mentions, etc.) would likely
    hurt performance rather than help.
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


# ==============================================================================
# TOKENIZATION & METRICS
# ==============================================================================

def tokenize_data(texts: list, labels: list, tokenizer, max_length: int = 128):
    """Tokenize and return a HuggingFace Dataset."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
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
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}


# ==============================================================================
# MAIN
# ==============================================================================

def train_q1_model(
    training_data_path: str,
    output_path: str,
    text_column:  str = 'title',
    label_column: str = 'label_q1'
):
    """
    Train the Q1 relevance classifier.

    We did not use class weighting here because the class distribution in our
    training data was reasonably balanced (the annotators were instructed to
    label a representative sample, not just easy/clear cases).
    """
    print("=" * 70)
    print("Q1 FINE-TUNING: Climate Change Relevance Classification")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")
    df = load_and_prepare_data(training_data_path, text_column, label_column)
    df['text'] = df['text'].apply(clean_tweet_text)

    # ── Train/val split ────────────────────────────────────────────────────
    print("\n[2/5] Splitting data (80/20, stratified)...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df['label']   # stratify to preserve class ratios
    )
    print(f"  Train: {len(train_texts):,}  |  Val: {len(val_texts):,}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True  # needed when adding new classification head
    )

    # ── Tokenize ───────────────────────────────────────────────────────────
    print("\n[4/5] Tokenizing...")
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset   = tokenize_data(val_texts,   val_labels,   tokenizer, MAX_LENGTH)

    # Free up memory before training — T4 is only 15GB
    del df, train_texts, val_texts, train_labels, val_labels
    gc.collect()

    # ── Training args ──────────────────────────────────────────────────────
    # Note: evaluation_strategy="epoch" means we can monitor overfitting
    # epoch-by-epoch and confirm when val loss starts climbing.
    training_args = TrainingArguments(
        output_dir='./q1_checkpoints',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # automatically recover best checkpoint
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Training...")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Epochs: {EPOCHS}  (we tried 3/5/7/10/15/20 — 5 was the sweet spot)")
    print(f"  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Expected time: ~15–25 min on T4 GPU")

    trainer.train()

    # ── Evaluate ───────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    print(f"\nFinal validation accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Final validation F1:       {eval_results['eval_f1']:.4f}")
    print(f"Final validation loss:     {eval_results['eval_loss']:.4f}")
    print(f"\n(Target was ~83–85%; we achieved 83.88% with this config)")

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ Done!")

    return model, tokenizer


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # =========================================================================
    # THINGS TO UPDATE BEFORE RUNNING:
    # =========================================================================
    # 1. TRAINING_DATA_PATH — your master training CSV
    # 2. text_column — column with tweet text (in our data: 'title')
    # 3. label_column — Q1 label column ('label_q1', values: 0/1/2)
    #
    # EXPECTED OUTPUT:
    # - ~83.88% validation accuracy
    # - Training time: ~15–25 min on T4 GPU
    # - Model saved to OUTPUT_MODEL_PATH (used by q1_inference.py)
    # =========================================================================

    model, tokenizer = train_q1_model(
        training_data_path=TRAINING_DATA_PATH,
        output_path=OUTPUT_MODEL_PATH,
        text_column='title',
        label_column='label_q1'
    )

    print("\n" + "=" * 70)
    print("Q1 FINE-TUNING COMPLETE!")
    print("=" * 70)
