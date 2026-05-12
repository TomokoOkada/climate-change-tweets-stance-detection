"""
Q4 Step 2 Inference: Human Responsibility to Act — Stance Classification
========================================================================
Classifies the stance toward human responsibility to act on climate change
for tweets identified as "Mentioned" in Q4 Step 1.

Classification Labels:
    0 (Against):       Explicitly states humans do NOT have a responsibility to act
    1 (For/Action):    Explicitly calls for action to mitigate climate change
    2 (Indeterminate): Tweet shows both stances, or stance is unclear

Input:  Q4 Step 1 inference output
Output: "Mentioned" tweets only + 3-class stance classification

Author: [Your Name]
Date: November 2025
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH      = "/kaggle/input/your-model/q4_step2_fine_tuned_model"
INPUT_DATA_PATH = "/kaggle/input/step2-inference-results/q4_step1_inference_results.csv"
OUTPUT_PATH     = "/kaggle/working/q4_step2_inference_results.csv"

BATCH_SIZE = 32
MAX_LENGTH = 128

LABEL_MAP = {
    0: "Against action",
    1: "For action",
    2: "Indeterminate"
}


# ==============================================================================
# DATASET
# ==============================================================================

class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts      = texts
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_filter_step1_results(filepath: str, text_column: str = 'title') -> pd.DataFrame:
    """
    Load Q4 Step 1 results and keep only "Mentioned" tweets for Step 2.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"✓ Loaded data with {enc} encoding")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not load {filepath}")

    print(f"  Total rows: {len(df):,}")

    if 'q4_step1_prediction' in df.columns:
        df = df[df['q4_step1_prediction'] == 'Mentioned'].copy()
        print(f"  Filtering to 'Mentioned': {len(df):,} tweets")
    elif 'q4_step1_prediction_numeric' in df.columns:
        df = df[df['q4_step1_prediction_numeric'] == 1].copy()
        print(f"  Filtering to 'Mentioned' (numeric): {len(df):,} tweets")
    else:
        print("  ⚠ Step 1 prediction column not found — using all rows")

    return df


# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(model, tokenizer, texts, batch_size, device):
    model.eval()
    dataset    = TweetDataset(texts, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Q4 Step 2"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            preds   = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


# ==============================================================================
# MAIN
# ==============================================================================

def main(model_path, input_path, output_path, text_column='title', batch_size=32):
    print("=" * 70)
    print("Q4 STEP 2 INFERENCE: Human Responsibility to Act — Stance Classification")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # BERTweet requires use_fast=False
    print(f"\n[1/4] Loading BERTweet model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    print(f"\n[2/4] Loading Step 1 results...")
    df    = load_and_filter_step1_results(input_path, text_column)
    texts = df[text_column].fillna('').astype(str).tolist()

    print(f"\n[3/4] Running inference on {len(texts):,} tweets...")
    predictions, probabilities = run_inference(model, tokenizer, texts, batch_size, device)

    df['q4_step2_prediction_numeric'] = predictions
    df['q4_step2_prediction']         = [LABEL_MAP[p] for p in predictions]
    df['q4_prob_against']             = probabilities[:, 0]
    df['q4_prob_for_action']          = probabilities[:, 1]
    df['q4_prob_indeterminate']       = probabilities[:, 2]
    df['q4_step2_confidence']         = probabilities.max(axis=1)

    print("\n[4/4] Results Summary:")
    print("-" * 50)
    for idx, label in LABEL_MAP.items():
        count = (predictions == idx).sum()
        pct   = count / len(predictions) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")
    print(f"\nMean confidence: {df['q4_step2_confidence'].mean():.4f}")

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Saved to {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")

    print("\n" + "=" * 70)
    print("Q4 STEP 2 INFERENCE COMPLETE!")
    print("=" * 70)
    return df


if __name__ == "__main__":
    # =========================================================================
    # CHECK POINTS FOR USER:
    # =========================================================================
    # 1. Update MODEL_PATH to your Q4 Step 2 model
    # 2. Update INPUT_DATA_PATH to your Q4 Step 1 inference output CSV
    # 3. BERTweet REQUIRES use_fast=False (already set in load call above)
    # 4. If tokenizer error: pip install transformers==4.44.2 tokenizers==0.19.1
    # 5. Output contains only "Mentioned" tweets with 3-class stance labels
    # =========================================================================

    results = main(
        model_path=MODEL_PATH,
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_PATH,
        text_column='title',
        batch_size=BATCH_SIZE
    )
