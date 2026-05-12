"""
Q2 Step 2 Inference: Climate Reality Stance — Classification
============================================================
Classifies the specific stance toward climate change reality for tweets
identified as "Mentioned" in Q2 Step 1.

Classification Labels:
    0 (Denial):        Explicitly states climate change is a hoax or fake
    1 (Acceptance):    Explicitly states climate change is real and causing effects
    2 (Indeterminate): Mixed stance or ambiguous

Input:  Q2 Step 1 results (all Q1-relevant tweets)
Output: "Mentioned" tweets only + 3-class stance labels

Model: roberta-large
    Note: The uploaded q2_step2_fine_tuning.py originally used BERTweet
    (vinai/bertweet-base) as a first attempt (81.03% accuracy). After
    testing multiple approaches, roberta-large gave the best result at
    83.85%. This inference script loads whichever model was saved by
    q2_step2_fine_tuning.py — make sure paths match.

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

MODEL_PATH = "/kaggle/input/your-model/q2_step2_model"
INPUT_DATA_PATH = "/kaggle/input/your-data/q2_step1_results.csv"
OUTPUT_PATH = "/kaggle/working/q2_step2_inference_results.csv"

BATCH_SIZE = 32
MAX_LENGTH = 128

LABEL_MAP = {
    0: "No / Denial",
    1: "Yes / Acceptance",
    2: "Indeterminate"
}


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class TweetDataset(Dataset):
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_filter_step1_results(
    filepath: str,
    text_column: str = 'title',
    step1_column: str = 'q2_step1_prediction'
) -> pd.DataFrame:
    """
    Load Q2 Step 1 results and filter to "Mentioned" only.
    
    Args:
        filepath: Path to Q2 Step 1 results
        text_column: Text column name
        step1_column: Step 1 prediction column
    
    Returns:
        Filtered DataFrame
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            break
        except:
            continue
    
    if df is None:
        raise ValueError(f"Could not load {filepath}")
    
    print(f"  Total tweets: {len(df):,}")
    
    # Filter to "Mentioned" only
    if step1_column in df.columns:
        # Handle both text and numeric
        if df[step1_column].dtype == 'object':
            mentioned_mask = df[step1_column] == "Mentioned"
        else:
            mentioned_mask = df[step1_column] == 1
    else:
        # Try index column
        mentioned_mask = df['q2_step1_prediction_index'] == 1
    
    df_filtered = df[mentioned_mask].copy()
    print(f"  Stance mentioned (for Step 2): {len(df_filtered):,}")
    
    return df_filtered


# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(model, tokenizer, texts, batch_size, device):
    """Run batch inference."""
    model.eval()
    
    dataset = TweetDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    return np.array(all_preds), all_probs


# ==============================================================================
# MAIN
# ==============================================================================

def main(model_path, input_path, output_path, text_column='title', batch_size=32):
    """Run Q2 Step 2 inference."""
    print("=" * 70)
    print("Q2 STEP 2 INFERENCE: Stance Classification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load model — roberta-large (final choice after testing BERTweet and others)
    # use_fast=True is fine for roberta-large; use_fast=False was only needed for BERTweet
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    print("✓ Model loaded")
    
    # Load data
    print("\n[2/4] Loading data...")
    df = load_and_filter_step1_results(input_path, text_column)
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Inference
    print("\n[3/4] Running inference...")
    predictions, probabilities = run_inference(
        model, tokenizer, texts, batch_size, device
    )
    
    # Add results
    df['q2_step2_prediction_numeric'] = predictions
    df['q2_step2_prediction'] = [LABEL_MAP[p] for p in predictions]
    df['prob_denial'] = probabilities[:, 0]
    df['prob_acceptance'] = probabilities[:, 1]
    df['prob_indeterminate'] = probabilities[:, 2]
    df['confidence'] = probabilities.max(axis=1)
    
    # Summary
    print("\n[4/4] Results Summary:")
    print("-" * 50)
    for idx, label in LABEL_MAP.items():
        count = (predictions == idx).sum()
        pct = count / len(predictions) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")
    
    print(f"\nMean confidence: {df['confidence'].mean():.4f}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print("✓ Saved!")
    
    print("\n" + "=" * 70)
    print("Q2 STEP 2 INFERENCE COMPLETE!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    # =========================================================================
    # THINGS TO UPDATE:
    # =========================================================================
    # 1. MODEL_PATH — your trained Q2 Step 2 model (roberta-large, 83.85%)
    # 2. INPUT_DATA_PATH — Q2 Step 1 inference output CSV
    # 3. text_column — tweet text column ('title' in our data)
    #
    # Note: This script uses roberta-large (not BERTweet).
    # If you saved a BERTweet model instead, change the tokenizer call to:
    #   AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # =========================================================================
    
    results = main(
        model_path=MODEL_PATH,
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_PATH,
        text_column='title',
        batch_size=BATCH_SIZE
    )
