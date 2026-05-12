"""
Q1 Inference: Climate Change Relevance Classification
======================================================
Runs inference on a large dataset of tweets using a fine-tuned RoBERTa model
to classify their relevance to climate change.

Classification Labels (from annotation codebook):
    0: NOT RELATED - The tweet is not related to climate change at all
    
    1: PARTIALLY RELATED - The tweet mentions climate change as one topic
       among other political/social issues
    
    2: PRIMARILY ABOUT CLIMATE CHANGE - The tweet is about climate change
       without equivalent references to other political issues

Input: CSV file containing tweets (typically ~360k original tweets)
Output: CSV file with original data + predicted labels + confidence scores

Downstream Usage:
    Tweets with labels 1 or 2 proceed to Q2-Q4 stance classification.
    Tweets with label 0 are excluded from further analysis.

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

# Paths - UPDATE THESE FOR YOUR ENVIRONMENT
MODEL_PATH = "/kaggle/input/your-model/q1_fine_tuned_model"
INPUT_DATA_PATH = "/kaggle/input/your-data/tweets.csv"
OUTPUT_PATH = "/kaggle/working/tweets_with_q1_predictions.csv"

# Inference settings
BATCH_SIZE = 32
MAX_LENGTH = 128

# Label mapping
LABEL_MAP = {
    0: "Not about climate change",
    1: "Partially about climate change",
    2: "Only about climate change"
}


# ==============================================================================
# CUSTOM DATASET CLASS
# ==============================================================================

class TweetDataset(Dataset):
    """
    PyTorch Dataset for batch processing of tweets.
    """
    
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of tweet texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
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

def load_tweets(filepath: str, text_column: str = 'title') -> pd.DataFrame:
    """
    Load tweet data from CSV file.
    
    Args:
        filepath: Path to CSV file
        text_column: Name of the column containing tweet text
    
    Returns:
        DataFrame with tweets
    """
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not load {filepath}")
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # Verify text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in data")
    
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning for inference.
    
    Args:
        text: Raw text
    
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================

def run_inference(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 32,
    device: str = "cuda"
) -> tuple:
    """
    Run batch inference on a list of texts.
    
    Args:
        model: Loaded classification model
        tokenizer: Corresponding tokenizer
        texts: List of texts to classify
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    
    # Create dataset and dataloader
    dataset = TweetDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    print(f"\nRunning inference on {len(texts):,} tweets...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(dataloader):,}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(
    model_path: str,
    input_path: str,
    output_path: str,
    text_column: str = 'title',
    batch_size: int = 32
):
    """
    Main function to run Q1 inference.
    
    Args:
        model_path: Path to fine-tuned model directory
        input_path: Path to input CSV with tweets
        output_path: Path to save results
        text_column: Name of text column in input CSV
        batch_size: Batch size for inference
    """
    print("=" * 70)
    print("Q1 INFERENCE: Climate Change Relevance Classification")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model and tokenizer
    print(f"\n[1/4] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load data
    print(f"\n[2/4] Loading tweets from {input_path}...")
    df = load_tweets(input_path, text_column)
    
    # Clean text
    df['clean_text'] = df[text_column].apply(clean_text)
    
    # Remove empty texts
    valid_mask = df['clean_text'].str.len() > 0
    df_valid = df[valid_mask].copy()
    print(f"  Valid tweets for inference: {len(df_valid):,}")
    
    # Run inference
    print(f"\n[3/4] Running inference...")
    predictions, probabilities = run_inference(
        model=model,
        tokenizer=tokenizer,
        texts=df_valid['clean_text'].tolist(),
        batch_size=batch_size,
        device=device
    )
    
    # Add predictions to dataframe
    df_valid['q1_prediction_index'] = predictions
    df_valid['q1_prediction_label'] = [LABEL_MAP[p] for p in predictions]
    df_valid['q1_confidence'] = probabilities.max(axis=1)
    
    # Show results summary
    print(f"\n[4/4] Results Summary:")
    print("-" * 50)
    print("\nPrediction Distribution:")
    for idx, label in LABEL_MAP.items():
        count = (predictions == idx).sum()
        pct = count / len(predictions) * 100
        print(f"  {idx} - {label}: {count:,} ({pct:.2f}%)")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {df_valid['q1_confidence'].mean():.4f}")
    print(f"  Min:  {df_valid['q1_confidence'].min():.4f}")
    print(f"  Max:  {df_valid['q1_confidence'].max():.4f}")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    df_valid.to_csv(output_path, index=False, encoding='utf-8')
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Results saved! ({file_size:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("Q1 INFERENCE COMPLETE!")
    print("=" * 70)
    
    return df_valid


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CHECK POINTS FOR USER:
    # =========================================================================
    # 1. Update MODEL_PATH to your fine-tuned Q1 model location
    # 2. Update INPUT_DATA_PATH to your tweet data CSV
    # 3. Verify the text column name (default: 'title')
    # 4. Adjust BATCH_SIZE based on your GPU memory (32-64 typical)
    # 5. Expected runtime: ~30-60 min for 360k tweets on T4 GPU
    # =========================================================================
    
    results = main(
        model_path=MODEL_PATH,
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_PATH,
        text_column='title',  # UPDATE if different in your data
        batch_size=BATCH_SIZE
    )
