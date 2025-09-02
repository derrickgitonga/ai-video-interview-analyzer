import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from models.nlp_model import NLPInterviewAnalyzer
from utils.text_preprocessing import TextPreprocessor
from utils.logger import setup_logging
import logging
import argparse
import os

setup_logging()
logger = logging.getLogger(__name__)

class InterviewTextDataset(Dataset):
    def __init__(self, texts, clarity_labels, sentiment_labels, keyword_labels, tokenizer, max_length=128):
        self.texts = texts
        self.clarity_labels = clarity_labels
        self.sentiment_labels = sentiment_labels
        self.keyword_labels = keyword_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clarity_label': torch.tensor(self.clarity_labels[idx], dtype=torch.float),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
            'keyword_labels': torch.tensor(self.keyword_labels[idx], dtype=torch.float)
        }

def generate_sample_data(num_samples=100):
    """Generate synthetic training data for demonstration"""
    logger.info("Generating synthetic training data...")
    
    sample_responses = [
        "I have extensive experience in machine learning and deep learning with PyTorch and TensorFlow",
        "My background includes natural language processing and computer vision applications",
        "I'm proficient in Python programming and have worked with cloud platforms like AWS",
        "I've deployed machine learning models in production environments using MLOps practices",
        "I have strong problem-solving skills and experience with containerization technologies",
        "While I have some AI experience, I'm more comfortable with traditional software development",
        "I'm excited about working on cutting-edge AI projects and learning new technologies",
        "I have research experience with PyTorch and production experience with TensorFlow",
        "I'm passionate about data science and enjoy solving complex business problems",
        "I have experience with both supervised and unsupervised learning algorithms"
    ]
    
    texts = []
    clarity_labels = []
    sentiment_labels = []
    keyword_labels = []
    
    for i in range(num_samples):
        # Select a random response template
        base_text = sample_responses[i % len(sample_responses)]
        
        # Add some variation
        variations = [
            "In my previous role, ",
            "Throughout my career, ",
            "I have demonstrated ",
            "My expertise includes ",
            "I successfully "
        ]
        
        text = variations[i % len(variations)] + base_text.lower()
        texts.append(text)
        
        # Generate labels
        clarity = np.random.uniform(0.6, 0.95)  # Most candidates have decent clarity
        clarity_labels.append(clarity)
        
        # Sentiment: 0=negative, 1=neutral, 2=positive
        sentiment = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])  # Mostly positive
        sentiment_labels.append(sentiment)
        
        # Keyword presence (binary for 10 keywords)
        keywords = np.random.randint(0, 2, 10).astype(float)
        # Increase probability of keyword matches for better responses
        if clarity > 0.8:
            keywords[:5] = 1  # First 5 keywords more likely
        keyword_labels.append(keywords)
    
    return texts, clarity_labels, sentiment_labels, keyword_labels

def train_nlp_model():
    parser = argparse.ArgumentParser(description='Train NLP Interview Analyzer')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--data_path', type=str, default='data/sample_interviews.csv')
    parser.add_argument('--save_path', type=str, default='models/nlp_model.pth')
    parser.add_argument('--num_samples', type=int, default=50)
    
    args = parser.parse_args()
    
    # Load and preprocess data
    logger.info("Loading training data...")
    
    texts = []
    clarity_labels = []
    sentiment_labels = []
    keyword_labels = []
    
    try:
        if os.path.exists(args.data_path):
            df = pd.read_csv(args.data_path)
            texts = df['response'].tolist()
            
            # Check if we have the required columns
            if 'clarity_score' in df.columns:
                clarity_labels = df['clarity_score'].tolist()
            else:
                clarity_labels = np.random.uniform(0.5, 1.0, len(texts)).tolist()
                
            if 'sentiment_label' in df.columns:
                sentiment_labels = df['sentiment_label'].tolist()
            else:
                sentiment_labels = np.random.randint(0, 3, len(texts)).tolist()
                
            # Extract keyword columns
            keyword_cols = [col for col in df.columns if col.startswith('keyword_')]
            if keyword_cols:
                keyword_labels = df[keyword_cols].values.tolist()
            else:
                keyword_labels = np.random.randint(0, 2, (len(texts), 10)).astype(float).tolist()
                
            logger.info(f"Loaded {len(texts)} samples from {args.data_path}")
        else:
            logger.warning(f"Data file {args.data_path} not found. Generating synthetic data...")
            texts, clarity_labels, sentiment_labels, keyword_labels = generate_sample_data(args.num_samples)
            
            # Save the generated data for future use
            os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            keyword_df = pd.DataFrame(keyword_labels, columns=[f'keyword_{i+1}' for i in range(10)])
            data_df = pd.DataFrame({
                'response': texts,
                'clarity_score': clarity_labels,
                'sentiment_label': sentiment_labels
            })
            data_df = pd.concat([data_df, keyword_df], axis=1)
            data_df.to_csv(args.data_path, index=False)
            logger.info(f"Saved synthetic data to {args.data_path}")
            
    except Exception as e:
        logger.error(f"Data loading failed: {e}. Generating synthetic data instead.")
        texts, clarity_labels, sentiment_labels, keyword_labels = generate_sample_data(args.num_samples)

    # Preprocess texts
    preprocessor = TextPreprocessor()
    cleaned_texts = [preprocessor.clean_text(text) for text in texts]

    # Split data
    train_texts, val_texts, train_clarity, val_clarity, train_sentiment, val_sentiment, train_keywords, val_keywords = train_test_split(
        cleaned_texts, clarity_labels, sentiment_labels, keyword_labels, 
        test_size=0.2, random_state=42
    )

    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = NLPInterviewAnalyzer(num_keywords=10)
    
    train_dataset = InterviewTextDataset(train_texts, train_clarity, train_sentiment, train_keywords, tokenizer)
    val_dataset = InterviewTextDataset(val_texts, val_clarity, val_sentiment, val_keywords, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Loss functions and optimizer
    clarity_criterion = nn.BCELoss()
    sentiment_criterion = nn.CrossEntropyLoss()
    keyword_criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Starting training on {device}...")
    logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        clarity_loss_total = 0
        sentiment_loss_total = 0
        keyword_loss_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            clarity_labels = batch['clarity_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            keyword_labels = batch['keyword_labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            clarity_loss = clarity_criterion(outputs['clarity'].squeeze(), clarity_labels)
            sentiment_loss = sentiment_criterion(outputs['sentiment'], sentiment_labels)
            keyword_loss = keyword_criterion(outputs['keywords'], keyword_labels)
            
            loss = clarity_loss + sentiment_loss + keyword_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            clarity_loss_total += clarity_loss.item()
            sentiment_loss_total += sentiment_loss.item()
            keyword_loss_total += keyword_loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_clarity_loss = 0
        val_sentiment_loss = 0
        val_keyword_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                clarity_labels = batch['clarity_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)
                keyword_labels = batch['keyword_labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                clarity_loss = clarity_criterion(outputs['clarity'].squeeze(), clarity_labels)
                sentiment_loss = sentiment_criterion(outputs['sentiment'], sentiment_labels)
                keyword_loss = keyword_criterion(outputs['keywords'], keyword_labels)
                
                val_loss += (clarity_loss + sentiment_loss + keyword_loss).item()
                val_clarity_loss += clarity_loss.item()
                val_sentiment_loss += sentiment_loss.item()
                val_keyword_loss += keyword_loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f} '
                   f'(Clarity: {clarity_loss_total/len(train_loader):.4f}, '
                   f'Sentiment: {sentiment_loss_total/len(train_loader):.4f}, '
                   f'Keyword: {keyword_loss_total/len(train_loader):.4f})')
        logger.info(f'  Val Loss: {avg_val_loss:.4f} '
                   f'(Clarity: {val_clarity_loss/len(val_loader):.4f}, '
                   f'Sentiment: {val_sentiment_loss/len(val_loader):.4f}, '
                   f'Keyword: {val_keyword_loss/len(val_loader):.4f})')
    
    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save_model(args.save_path)
    logger.info(f"Training completed. Model saved to {args.save_path}")

if __name__ == '__main__':
    train_nlp_model()