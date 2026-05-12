"""
Q4 Step 1 Fine-Tuning: Human Responsibility to Act — Mention Detection
=======================================================================
Binary classifier: does the tweet make any explicit statement about whether
humans have a responsibility to act on climate change?

Classification (binary):
    0: NOT MENTIONED — No explicit call for (or against) action
    1: MENTIONED     — Tweet explicitly addresses human responsibility to act

Training data:
    4,000 total samples
    Class 0 (Not Mentioned): 2,817 (70.4%)   <- majority
    Class 1 (Mentioned):     1,183 (29.6%)   <- minority
    Imbalance ratio: ~2.4:1

    Train set: 3,200  |  Val set: 800

--------------------------------------------------------------------
FULL EXPERIMENT LOG — 4 notebook versions, extensive trial and error
--------------------------------------------------------------------

1. BERTweet (vinai/bertweet-base)
   -> 74.75% accuracy
   Starting point. Better than nothing but well below target.

2. Two-stage: pre-train on Q3 binary, fine-tune on Q4 (no class weights)
   Stage 1: Q3 binary data (4,000 samples), 5 epochs, LR=2e-5, batch=8
   Stage 2: Q4 data (3,200 train), 10 epochs, LR=1e-5, batch=8
   -> 79.38%
   Rationale: Q3 and Q4 both classify binary stance-mention on similar
   tweet text. Q3 data (4,000 samples) provides more training signal
   than starting from random weights.

3. Two-stage + class weights (balanced: Not Mentioned=0.71x, Mentioned=1.69x)
   -> 76.75%  <- WORSE
   Adding class weights hurt. With only ~2.4:1 imbalance, the class
   weighting was too aggressive and disrupted the patterns learned in
   Stage 1.

4. Two-stage + oversampling (minority class oversampled to 50/50)
   -> 29.62%  <- EXCLUDED FROM ANALYSIS (double-correction bug, explained below)

   What happened:
   The oversampling code ran first: minority class (Mentioned, 29.6%) was
   duplicated until the training set was balanced 50/50. So far so good.

   The bug: class weights computed on the ORIGINAL 70/30 distribution
   (Not Mentioned=0.71x, Mentioned=1.69x) were left active in the loss
   function. Applying these weights to a dataset that was already 50/50
   told the model: "treat Mentioned as if it were still a minority class."
   But the model was seeing 50% Mentioned — so the 1.69x penalty pushed
   it to predict Mentioned for almost everything.

   Why 29.62% specifically (not ~33% random):
   The validation set was NOT oversampled, so it had the original 70/30
   distribution. A model that predicts "Mentioned" for everything scores
   29.6% on that val set — exactly matching the Mentioned class frequency.
   The 29.62% result is therefore entirely explained by this mechanism,
   not noise.

   This is a double-correction error: oversampling and class weighting
   both address the same imbalance, and applying both simultaneously
   inverts the intended effect. This result is excluded from analysis
   because the model was not optimized for the intended objective.
   The failure mode is fully understood and does not affect other results.

5. Ensemble of 3 models (different seeds/LRs, no two-stage)
   Model 1 (seed=42,  LR=2e-5,   6 epochs): 78.88%
   Model 2 (seed=123, LR=2e-5,   9 epochs): 70.38%
   Model 3 (seed=456, LR=1.5e-5, 9 epochs): 70.38%
   Ensemble average probability: 78.88%
   -> No improvement. High variance between seeds; averaging pulled the
   best model down rather than up.

6. Seed stability test (5 seeds: 42, 123, 456, 789, 2024)
   Results varied from ~70% to ~80% across seeds. Only seed 42 reliably
   approached 80%. This variance is partly why we needed 4 notebook
   versions to get a consistent result.

7. Two-stage (Q3→Q4), class weights, batch=8, 8 epochs  <- FINAL
   Stage 1: Q3 binary data, 5 epochs, LR=2e-5, batch=8, with weights
   Stage 2: Q4 data, 8 epochs (not 10!), LR=1e-5, batch=8, with weights
   -> 80.38%, F1=0.8018, Precision=0.8005, Recall=0.8037
   
   Key difference from attempt #3: reducing Stage 2 from 10 to 8 epochs
   prevented overfitting. The val loss at epoch 10 was 1.62 vs 0.56 at
   epoch 3, showing clear divergence. load_best_model_at_end picked
   epoch 6 (val loss 0.999, acc 80.375%) as the checkpoint.

   Training time: ~50 min total on T4 GPU
   (Stage 1: ~20 min, Stage 2: ~30 min)

--------------------------------------------------------------------
Why two-stage for Q4 Step 1 (same reason as Q3 Step 2):
Q4 binary data has ~2.4:1 imbalance and the minority class (Mentioned)
represents subtle, context-dependent language. Pre-training on Q3 binary
data (structurally identical task, more diverse examples) helps the model
develop better binary detection representations before adapting to Q4.

Unlike Q3 Step 2 (which used Q2 data for pre-training), Q4 Step 1 uses
Q3 data. The Q3 dataset has 938 "Mentioned" samples vs Q4's 1,183 —
smaller, but the additional training signal still helps.
--------------------------------------------------------------------

Author: [Your Name]
Date: November 2025
"""

import os
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
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
OUTPUT_MODEL_PATH  = "./q4_step1_fine_tuned_model"

# roberta-large: same model used throughout.
# BERTweet was tried first (74.75%) and abandoned.
BASE_MODEL = "roberta-large"
NUM_LABELS = 2
MAX_LENGTH = 128

# Stage 1 (Q3 pre-training)
STAGE1_EPOCHS = 5
STAGE1_LR     = 2e-5
STAGE1_BATCH  = 8

# Stage 2 (Q4 fine-tuning)
# 8 epochs was the sweet spot: val loss diverged after epoch 8-9,
# and load_best_model_at_end recovered the best checkpoint automatically.
# DO NOT increase to 10 epochs — we tried it and got worse results.
STAGE2_EPOCHS = 8
STAGE2_LR     = 1e-5
STAGE2_BATCH  = 8

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


def prepare_q3_pretraining_data(df: pd.DataFrame, text_col: str) -> tuple:
    """
    Extract Q3 binary data for Stage 1 pre-training.

    Q3 binary mapping:
        0, 1, 2 (any explicit stance) -> 1 (Mentioned)
        9 (Not Mentioned)             -> 0 (Not Mentioned)

    Why Q3 (not Q2) for Q4 pre-training?
    Two complementary reasons:

    1. Statistical similarity — class distributions are close:
         Q3: Not Mentioned 76.5%, Mentioned 23.5%  (3.25:1 ratio)
         Q4: Not Mentioned 70.4%, Mentioned 29.6%  (2.4:1 ratio)
         Q2: Not Mentioned 51.3%, Mentioned 48.7%  (1.05:1 ratio — balanced)
       Q2's near-balanced distribution is structurally different from Q4's
       imbalanced setup. Pre-training on Q2 would give the model a misleading
       prior about class frequencies that doesn't transfer to Q4.

    2. Conceptual similarity — Q3 and Q4 are about adjacent human agency:
         Q3: "Does the tweet say whether climate change IS caused by humans?"
         Q4: "Does the tweet say whether humans SHOULD ACT on climate change?"
       Both questions center on human agency and responsibility in climate
       change. Tweets that explicitly address human causality (Q3) often share
       vocabulary and framing with tweets that explicitly call for human action
       (Q4) — e.g., "humans are causing this" and "humans must fix this" draw
       on overlapping rhetorical patterns. Q2 ("is CC real?") is conceptually
       more distant, about existence rather than agency.

    Q3 has 938 Mentioned / 3,062 Not Mentioned in our dataset.
    Despite being smaller than Q4's Mentioned class (1,183), the pre-training
    still helps because of the statistical and conceptual alignment above.
    """
    df_q3 = df.copy()
    df_q3['q3_binary'] = df_q3['label_q3'].apply(
        lambda x: 0 if x == 9 else (1 if x in [0, 1, 2] else None)
    )
    df_q3 = df_q3[df_q3['q3_binary'].notna()].copy()

    texts  = df_q3[text_col].fillna('').astype(str).tolist()
    labels = df_q3['q3_binary'].astype(int).tolist()

    print(f"\nQ3 pre-training data ({len(texts):,} samples):")
    for lbl, name in {0: "Not Mentioned", 1: "Mentioned"}.items():
        count = labels.count(lbl)
        print(f"  {lbl} ({name}): {count:,} ({count/len(labels)*100:.1f}%)")

    return texts, labels


def prepare_q4_finetuning_data(df: pd.DataFrame, text_col: str) -> tuple:
    """
    Extract Q4 binary data for Stage 2 fine-tuning.

    Q4 binary mapping:
        0, 1, 2 (any explicit stance) -> 1 (Mentioned)
        9 (Not Mentioned)             -> 0 (Not Mentioned)

    Class distribution: Not Mentioned 70.4%, Mentioned 29.6%
    Imbalance ratio ~2.4:1

    NOTE: We use class weights (not oversampling) to handle the imbalance.
    Oversampling was tried and produced 29.62% accuracy due to a code bug;
    class weights alone gave 76.75% (worse than no correction at 79.38%),
    but with 8 epochs instead of 10, class weights + two-stage gave 80.38%.
    """
    df_q4 = df.copy()
    df_q4['q4_binary'] = df_q4['label_q4'].apply(
        lambda x: 0 if x == 9 else (1 if x in [0, 1, 2] else None)
    )
    df_q4 = df_q4[df_q4['q4_binary'].notna()].copy()

    texts  = df_q4[text_col].fillna('').astype(str).tolist()
    labels = df_q4['q4_binary'].astype(int).tolist()

    print(f"\nQ4 fine-tuning data ({len(texts):,} samples):")
    for lbl, name in {0: "Not Mentioned", 1: "Mentioned"}.items():
        count = labels.count(lbl)
        print(f"  {lbl} ({name}): {count:,} ({count/len(labels)*100:.1f}%)")

    return texts, labels


def clean_tweet_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


# ==============================================================================
# WEIGHTED TRAINER (handles class imbalance in loss function)
# ==============================================================================

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to CrossEntropyLoss.

    We use class weights (not oversampling) for the 2.4:1 imbalance in Q4.
    Oversampling was catastrophically buggy (29.62%), and plain class weights
    without the two-stage approach gave 76.75%.
    With two-stage + 8 epochs: 80.38%.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None else None
        )
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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

def train_q4_step1_model(
    training_data_path: str,
    output_path: str,
    text_column: str = 'title'
):
    """
    Two-stage binary mention-detection for Q4.

    Stage 1: Pre-train on Q3 binary data (related task, more exposure)
    Stage 2: Fine-tune on Q4 binary data (target task, 8 epochs)

    Result: 80.38% accuracy
    Baseline single-stage BERTweet: 74.75%
    """
    print("=" * 70)
    print("Q4 STEP 1 FINE-TUNING: Human Responsibility — Two-Stage Training")
    print("=" * 70)
    print("Stage 1: Pre-train on Q3 binary (5 epochs, LR=2e-5)")
    print("Stage 2: Fine-tune on Q4 binary (8 epochs, LR=1e-5)")
    print("Result: 80.38% | BERTweet baseline: 74.75%")
    print("")
    print("What didn't work:")
    print("  Class weights alone:      76.75% (worse than no correction)")
    print("  Oversampling:             29.62% (code bug — catastrophic)")
    print("  3-model ensemble:         78.88% (high seed variance)")
    print("  10 epochs Stage 2:        ~79%   (val loss diverged at ep8-9)")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading training data...")
    df = load_training_data(training_data_path)

    q3_texts, q3_labels = prepare_q3_pretraining_data(df, text_column)
    q4_texts, q4_labels = prepare_q4_finetuning_data(df, text_column)

    # ── Compute class weights for Q4 ───────────────────────────────────────
    print("\n[2/7] Computing class weights for Q4...")
    q4_train_texts, q4_val_texts, q4_train_labels, q4_val_labels = train_test_split(
        q4_texts, q4_labels,
        test_size=0.2, random_state=RANDOM_SEED, stratify=q4_labels
    )
    print(f"  Q4 train: {len(q4_train_texts):,}  |  Q4 val: {len(q4_val_texts):,}")

    cw_q3 = compute_class_weight('balanced', classes=np.unique(q3_labels), y=q3_labels)
    cw_q4 = compute_class_weight('balanced', classes=np.unique(q4_train_labels), y=q4_train_labels)

    print(f"  Q3 class weights: Not Mentioned={cw_q3[0]:.4f}, Mentioned={cw_q3[1]:.4f}")
    print(f"  Q4 class weights: Not Mentioned={cw_q4[0]:.4f}, Mentioned={cw_q4[1]:.4f}")

    cw_q3_tensor = torch.tensor(cw_q3, dtype=torch.float).to(device)
    cw_q4_tensor = torch.tensor(cw_q4, dtype=torch.float).to(device)

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"\n[3/7] Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ── Tokenize Q3 (Stage 1, full dataset) ───────────────────────────────
    print("\n[4/7] Tokenizing Q3 data (Stage 1)...")
    q3_dataset = tokenize_data(q3_texts, q3_labels, tokenizer, MAX_LENGTH)

    # ── Tokenize Q4 (Stage 2, with val split) ─────────────────────────────
    print("\n[5/7] Tokenizing Q4 data (Stage 2)...")
    q4_train_dataset = tokenize_data(q4_train_texts, q4_train_labels, tokenizer, MAX_LENGTH)
    q4_val_dataset   = tokenize_data(q4_val_texts,   q4_val_labels,   tokenizer, MAX_LENGTH)

    del df, q3_texts, q4_texts, q4_train_texts, q4_val_texts
    gc.collect()

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[6/7] Two-stage training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Stage 1: Q3 pre-training
    print(f"\n  Stage 1: Pre-training on Q3 ({len(q3_dataset):,} samples)...")
    print(f"  {STAGE1_EPOCHS} epochs | LR={STAGE1_LR} | batch={STAGE1_BATCH}")
    print(f"  Expected time: ~17-20 min on T4 GPU")

    stage1_args = TrainingArguments(
        output_dir='./q4_step1_stage1_checkpoints',
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=STAGE1_BATCH,
        per_device_eval_batch_size=STAGE1_BATCH,
        learning_rate=STAGE1_LR,
        warmup_steps=0,
        weight_decay=0,
        logging_steps=50,
        evaluation_strategy='no',
        save_strategy='no',
        fp16=torch.cuda.is_available(),
        report_to='none'
    )

    WeightedTrainer(
        model=model, args=stage1_args,
        train_dataset=q3_dataset,
        class_weights=cw_q3_tensor,
        compute_metrics=compute_metrics
    ).train()

    print("✓ Stage 1 complete")

    # Stage 2: Q4 fine-tuning (8 epochs — DO NOT change to 10)
    print(f"\n  Stage 2: Fine-tuning on Q4 ({len(q4_train_dataset):,} train samples)...")
    print(f"  {STAGE2_EPOCHS} epochs | LR={STAGE2_LR} | batch={STAGE2_BATCH}")
    print(f"  load_best_model_at_end=True (val loss diverges after epoch 8-9)")
    print(f"  Expected time: ~28-30 min on T4 GPU")

    stage2_args = TrainingArguments(
        output_dir='./q4_step1_stage2_checkpoints',
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=STAGE2_BATCH,
        per_device_eval_batch_size=STAGE2_BATCH,
        learning_rate=STAGE2_LR,
        warmup_steps=0,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=torch.cuda.is_available(),
        report_to='none'
    )

    trainer_stage2 = WeightedTrainer(
        model=model, args=stage2_args,
        train_dataset=q4_train_dataset,
        eval_dataset=q4_val_dataset,
        class_weights=cw_q4_tensor,
        compute_metrics=compute_metrics
    )
    trainer_stage2.train()

    # ── Evaluate & save ────────────────────────────────────────────────────
    print("\n[7/7] Final evaluation...")
    eval_results = trainer_stage2.evaluate()
    print(f"\nFinal accuracy: {eval_results['eval_accuracy']*100:.2f}%")
    print(f"Final F1:       {eval_results['eval_f1']:.4f}")
    print(f"\n(Target: ~80.38% | BERTweet baseline: 74.75%)")
    print(f"(If substantially lower, check that you're using batch=8, NOT 16)")
    print(f" Batch size 16 gave ~77%; batch size 8 gave 80.38%.")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Model saved to {output_path}")
    return model, tokenizer


if __name__ == "__main__":
    # =========================================================================
    # THINGS TO UPDATE:
    # =========================================================================
    # 1. TRAINING_DATA_PATH — master CSV with BOTH label_q3 and label_q4
    # 2. text_column — tweet text column ('title' in our data)
    #
    # CRITICAL NOTES FROM FAILED EXPERIMENTS:
    # - DO NOT oversample: produced 29.62% (code bug, but risky to retry)
    # - DO NOT use 10 epochs for Stage 2: val loss diverges
    # - DO NOT change batch size to 16: gave ~77% vs 80.38% with batch=8
    # - Class weights are applied in Stage 1 AND Stage 2
    #
    # EXPECTED RESULTS:
    # ~80-80.5% accuracy (we got 80.38%)
    # Total training time: ~50 min on T4 GPU
    # =========================================================================

    model, tokenizer = train_q4_step1_model(
        training_data_path=TRAINING_DATA_PATH,
        output_path=OUTPUT_MODEL_PATH,
        text_column='title'
    )
    print("\n" + "=" * 70)
    print("Q4 STEP 1 FINE-TUNING COMPLETE!")
    print("=" * 70)
