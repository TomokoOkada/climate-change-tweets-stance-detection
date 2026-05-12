"""
Q2 Step 1 Inference: Stance Mentioned Classification
=====================================================
Run inference to classify whether tweets mention a stance on climate change
being real/happening.

Input: Tweets with Q1 predictions (filtered to Q1=1 or Q1=2)
Output: Tweets with Q2 Step 1 predictions (Mentioned/Not Mentioned)

Classification Labels:
    0: Not Mentioned - Tweet doesn't express belief about climate reality
    1: Mentioned - Tweet expresses belief about climate change being real

Author: [Your Name]
Date: October 2025
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

MODEL_PATH = "/kaggle/input/your-model/q2_step1_model"
INPUT_DATA_PATH = "/kaggle/input/your-data/q1_inference_results.csv"
OUTPUT_PATH = "/kaggle/working/q2_step1_inference_results.csv"

BATCH_SIZE = 64
MAX_LENGTH = 128

LABEL_MAP = {
    0: "Not Mentioned",
    1: "Mentioned"
}


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class TweetDataset(Dataset):
    """PyTorch Dataset for tweet classification."""
    
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
# DATA LOADING AND FILTERING
# ==============================================================================

def load_and_filter_q1_results(
    filepath: str,
    text_column: str = 'title',
    q1_prediction_column: str = 'q1_prediction_index'
) -> pd.DataFrame:
    """
    Load Q1 results and filter to climate-related tweets only.
    
    Only tweets with Q1=1 (partially about climate) or Q1=2 (only about climate)
    should be classified in Q2.
    
    Args:
        filepath: Path to Q1 inference results
        text_column: Column with tweet text
        q1_prediction_column: Column with Q1 predictions (0, 1, or 2)
    
    Returns:
        Filtered DataFrame
    """
    # Load with encoding handling
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            print(f"✓ Loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not load {filepath}")
    
    print(f"  Total tweets: {len(df):,}")
    
    # Filter to climate-related tweets (Q1 = 1 or 2)
    climate_mask = df[q1_prediction_column].isin([1, 2])
    df_filtered = df[climate_mask].copy()
    
    print(f"  Climate-related (Q1=1 or 2): {len(df_filtered):,}")
    print(f"  Filtered out (Q1=0): {len(df) - len(df_filtered):,}")
    
    return df_filtered


# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 64,
    device: str = "cuda"
) -> tuple:
    """
    Run batch inference.
    
    Args:
        model: Classification model
        tokenizer: Tokenizer
        texts: List of texts
        batch_size: Batch size
        device: Device (cuda/cpu)
    
    Returns:
        Tuple of (predictions, confidences)
    """
    model.eval()
    
    dataset = TweetDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_confidences = []
    
    print(f"\nRunning inference on {len(texts):,} tweets...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_confidences.extend(probs.max(dim=1).values.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_confidences)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(
    model_path: str,
    input_path: str,
    output_path: str,
    text_column: str = 'title',
    batch_size: int = 64
):
    """
    Run Q2 Step 1 inference.
    
    Args:
        model_path: Path to fine-tuned model
        input_path: Path to Q1 results CSV
        output_path: Path to save Q2 Step 1 results
        text_column: Text column name
        batch_size: Batch size
    """
    print("=" * 70)
    print("Q2 STEP 1 INFERENCE: Stance Mentioned Classification")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load and filter data
    print(f"\n[2/4] Loading and filtering data...")
    df = load_and_filter_q1_results(input_path, text_column)
    
    # Prepare texts
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Run inference
    print(f"\n[3/4] Running inference...")
    predictions, confidences = run_inference(
        model, tokenizer, texts, batch_size, device
    )
    
    # Add results
    df['q2_step1_prediction_index'] = predictions
    df['q2_step1_prediction'] = [LABEL_MAP[p] for p in predictions]
    df['q2_step1_confidence'] = confidences
    
    # Summary
    print(f"\n[4/4] Results Summary:")
    print("-" * 50)
    for idx, label in LABEL_MAP.items():
        count = (predictions == idx).sum()
        pct = count / len(predictions) * 100
        print(f"  {label}: {count:,} ({pct:.2f}%)")
    
    print(f"\nMean confidence: {confidences.mean():.4f}")
    
    # Count how many will go to Step 2
    mentioned_count = (predictions == 1).sum()
    print(f"\n→ Tweets for Q2 Step 2: {mentioned_count:,}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Saved!")
    
    print("\n" + "=" * 70)
    print("Q2 STEP 1 INFERENCE COMPLETE!")
    print("=" * 70)
    
    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CHECK POINTS FOR USER:
    # =========================================================================
    # 1. Update MODEL_PATH to your Q2 Step 1 fine-tuned model
    # 2. Update INPUT_DATA_PATH to your Q1 inference results
    # 3. Verify Q1 results have 'q1_prediction_index' column
    # 4. Only Q1=1 or Q1=2 tweets will be processed
    # 5. Output: "Mentioned" tweets will need Q2 Step 2 classification
    # =========================================================================
    
    results = main(
        model_path=MODEL_PATH,
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_PATH,
        text_column='title',
        batch_size=BATCH_SIZE
    )
