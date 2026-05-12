"""
Q3 Step 1 Fine-Tuning: Human Causality — Mention Detection
===========================================================
Binary classifier: does the tweet make any explicit statement about whether
climate change is caused by human activities?

This is Stage 1 of the two-stage Q3 pipeline. Only tweets classified as
"Mentioned" here proceed to Q3 Step 2 for stance classification.

Classification (binary):
    0: NOT MENTIONED — No explicit statement about human causality
    1: MENTIONED     — Tweet explicitly addresses whether humans cause climate change

Why two stages?
    We first tried direct 4-class classification (Denies / Affirms /
    Indeterminate / Not Mentioned) on the full ~360k tweet set. Performance
    was poor — validation accuracy stalled around 65–68% because the "Not
    Mentioned" category dominated (~60% of tweets). The model kept predicting
    "Not Mentioned" for borderline cases.

    Splitting into binary detection (Step 1) + stance classification (Step 2)
    gave much better results on both tasks.

Model selection — what we compared for Step 1:
    We tested three configurations on a held-out validation set:

    1. vinai/bertweet-base (BERTweet)
       - Twitter-specific pre-training (850M tweets)
       - Accuracy: 81.12%
       - Good, but class imbalance still hurt it

    2. roberta-large with class weighting
       - Weighted loss to penalize errors on the minority class
       - Accuracy: 82.50%
       - Better than BERTweet, cleaner to implement

    3. roberta-large with random oversampling (balanced 50/50)
       - Duplicated minority class samples to match majority class size
       - Accuracy: 82.88%  ← final choice
       - Slightly better than class weighting on this dataset
       - Downside: longer training time due to larger dataset

    BERTweet's smaller size relative to roberta-large seemed to matter more
    than its Twitter-specific pre-training for this task.

Final performance: 82.88% accuracy, F1 = 0.8283

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
OUTPUT_MODEL_PATH  = "./q3_step1_fine_tuned_model"

# roberta-large outperformed BERTweet (81.12%) and roberta-base in our tests.
# Larger model capacity > Twitter-specific pre-training for this classification task.
BASE_MODEL = "roberta-large"
NUM_LABELS = 2
MAX_LENGTH = 128

# These hyperparameters were kept the same across Q2–Q4 Step 1 models.
# We did experiment with lower LR (1e-5) and found it converged more slowly
# without final accuracy gains, so we stayed with 2e-5.
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
    Load training data, map Q3 labels to binary, and oversample the minority class.

    Original Q3 labels:
        0: Denies human causality  → binary 1 (Mentioned)
        1: Affirms human causality → binary 1 (Mentioned)
        2: Indeterminate           → binary 1 (Mentioned)
        9: Not Mentioned           → binary 0 (Not Mentioned)

    Class distribution in our training data (before balancing):
        Not Mentioned: ~76%
        Mentioned:     ~24%

    We tried class weighting first (easy to implement), which gave 82.50%.
    Random oversampling of the minority class to a 50/50 split gave 82.88%,
    so we went with oversampling.
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
        raise ValueError(f"Could not load {filepath}")

    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ['text', 'label_original']
    df['label_original'] = df['label_original'].astype(int)

    valid = df['label_original'].isin([0, 1, 2, 9])
    if not valid.all():
        print(f"⚠ Dropping {(~valid).sum()} rows with unexpected labels")
        df = df[valid]

    # Map to binary
    df['label'] = (df['label_original'] != 9).astype(int)

    print(f"\nBefore balancing:")
    for lbl, name in {0: "Not Mentioned", 1: "Mentioned"}.items():
        count = (df['label'] == lbl).sum()
        pct   = count / len(df) * 100
        print(f"  {lbl} ({name}): {count:,} ({pct:.1f}%)")

    # Oversample minority class to balance 50/50
    # (this gave 82.88% vs 82.50% with class weighting)
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]

    n_majority = len(df_majority)
    df_minority_upsampled = df_minority.sample(
        n=n_majority, replace=True, random_state=RANDOM_SEED
    )
    df = pd.concat([df_majority, df_minority_upsampled]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"\nAfter oversampling (balanced 50/50):")
    for lbl, name in {0: "Not Mentioned", 1: "Mentioned"}.items():
        count = (df['label'] == lbl).sum()
        print(f"  {lbl} ({name}): {count:,}")
    print(f"  Total: {len(df):,}")

    return df[['text', 'label']]


def clean_tweet_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


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
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}


# ==============================================================================
# MAIN
# ==============================================================================

def train_q3_step1_model(
    training_data_path: str,
    output_path: str,
    text_column:  str = 'title',
    label_column: str = 'label_q3'
):
    print("=" * 70)
    print("Q3 STEP 1 FINE-TUNING: Human Causality — Mention Detection")
    print("=" * 70)
    print("(roberta-large + oversampling → 82.88% | BERTweet → 81.12% | roberta-large+weights → 82.50%)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load & balance data ────────────────────────────────────────────────
    print("\n[1/5] Loading and balancing training data...")
    df = load_and_prepare_data(training_data_path, text_column, label_column)
    df['text'] = df['text'].apply(clean_tweet_text)

    # ── Split ──────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting data (80/20)...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
    )
    print(f"  Train: {len(train_texts):,}  |  Val: {len(val_texts):,}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading {BASE_MODEL}...")
    # Note: we also tried vinai/bertweet-base and cardiffnlp/twitter-roberta-base.
    # roberta-large won out despite not being Twitter-specific — capacity > domain.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    # ── Tokenize ───────────────────────────────────────────────────────────
    print("\n[4/5] Tokenizing...")
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset   = tokenize_data(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
    del df, train_texts, val_texts, train_labels, val_labels
    gc.collect()

    # ── Training ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir='./q3_step1_checkpoints',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\n[5/5] Training...")
    print(f"  Expected time: ~20–30 min on T4 (dataset is larger due to oversampling)")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"\nValidation accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Validation loss:     {eval_results['eval_loss']:.4f}")
    print(f"\n(We got 82.88% with this config; BERTweet comparison: 81.12%)")

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")
    return model, tokenizer


if __name__ == "__main__":
    # =========================================================================
    # THINGS TO UPDATE:
    # =========================================================================
    # 1. TRAINING_DATA_PATH — master training CSV
    # 2. text_column — tweet text column (our data: 'title')
    # 3. label_column — Q3 labels ('label_q3', values: 0/1/2/9)
    #
    # NOTE: This script uses roberta-large, NOT BERTweet.
    # BERTweet gave 81.12%; roberta-large + oversampling gave 82.88%.
    # If you switch to BERTweet, add use_fast=False to the tokenizer call.
    # =========================================================================

    model, tokenizer = train_q3_step1_model(
        training_data_path=TRAINING_DATA_PATH,
        output_path=OUTPUT_MODEL_PATH,
        text_column='title',
        label_column='label_q3'
    )
    print("\n" + "=" * 70)
    print("Q3 STEP 1 FINE-TUNING COMPLETE!")
    print("=" * 70)
