import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.vision_model import VisionInterviewAnalyzer
from utils.video_preprocessing import VideoPreprocessor
from utils.logger import setup_logging
import logging
import argparse
import os

setup_logging()
logger = logging.getLogger(__name__)

class InterviewVideoDataset(Dataset):
    def __init__(self, video_paths, engagement_labels, confidence_labels, attention_labels, preprocessor, seq_length=30):
        self.video_paths = video_paths
        self.engagement_labels = engagement_labels
        self.confidence_labels = confidence_labels
        self.attention_labels = attention_labels
        self.preprocessor = preprocessor
        self.seq_length = seq_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # For demo purposes, generate synthetic frames if video file doesn't exist
        if not os.path.exists(video_path):
            logger.warning(f"Video file {video_path} not found. Generating synthetic data.")
            # Generate random frames as placeholder
            frames = torch.randn(self.seq_length, 3, 224, 224)
        else:
            frames = self.preprocessor.preprocess_video(video_path, self.seq_length)
            
            if frames is None:
                # Return dummy data if video processing fails
                frames = torch.randn(self.seq_length, 3, 224, 224)
        
        return {
            'frames': frames,
            'engagement_label': torch.tensor(self.engagement_labels[idx], dtype=torch.float),
            'confidence_label': torch.tensor(self.confidence_labels[idx], dtype=torch.float),
            'attention_label': torch.tensor(self.attention_labels[idx], dtype=torch.float)
        }

def generate_sample_video_data(num_samples=20):
    """Generate synthetic video metadata for demonstration"""
    logger.info("Generating synthetic video metadata...")
    
    video_paths = [f"data/sample_video_{i+1}.mp4" for i in range(num_samples)]
    
    # Generate realistic scores
    engagement_labels = np.random.uniform(0.6, 0.95, num_samples)
    confidence_labels = np.random.uniform(0.5, 0.9, num_samples)
    attention_labels = np.random.uniform(0.7, 0.98, num_samples)
    
    return video_paths, engagement_labels, confidence_labels, attention_labels

def train_vision_model():
    parser = argparse.ArgumentParser(description='Train Vision Interview Analyzer')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--data_csv', type=str, default='data/video_metadata.csv')
    parser.add_argument('--save_path', type=str, default='models/vision_model.pth')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seq_length', type=int, default=15)
    
    args = parser.parse_args()
    
    # Load video metadata
    logger.info("Loading training data...")
    
    video_paths = []
    engagement_labels = []
    confidence_labels = []
    attention_labels = []
    
    try:
        if os.path.exists(args.data_csv):
            df = pd.read_csv(args.data_csv)
            video_paths = df['video_path'].tolist()
            
            # Extract scores
            engagement_labels = df['engagement_score'].tolist() if 'engagement_score' in df.columns else []
            confidence_labels = df['confidence_score'].tolist() if 'confidence_score' in df.columns else []
            attention_labels = df['attention_score'].tolist() if 'attention_score' in df.columns else []
            
            logger.info(f"Loaded {len(video_paths)} samples from {args.data_csv}")
        else:
            logger.warning(f"Data file {args.data_csv} not found. Generating synthetic data...")
            video_paths, engagement_labels, confidence_labels, attention_labels = generate_sample_video_data(args.num_samples)
            
            # Save the generated metadata
            os.makedirs(os.path.dirname(args.data_csv), exist_ok=True)
            data_df = pd.DataFrame({
                'video_path': video_paths,
                'engagement_score': engagement_labels,
                'confidence_score': confidence_labels,
                'attention_score': attention_labels
            })
            data_df.to_csv(args.data_csv, index=False)
            logger.info(f"Saved synthetic video metadata to {args.data_csv}")
            
    except Exception as e:
        logger.error(f"Data loading failed: {e}. Generating synthetic data instead.")
        video_paths, engagement_labels, confidence_labels, attention_labels = generate_sample_video_data(args.num_samples)

    # Initialize model and preprocessor
    preprocessor = VideoPreprocessor()
    model = VisionInterviewAnalyzer()
    
    # Create datasets
    train_dataset = InterviewVideoDataset(
        video_paths, engagement_labels, confidence_labels, attention_labels, 
        preprocessor, args.seq_length
    )
    
    # Use a simple train/val split (for demo, we'll use all data for training)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Starting training on {device}...")
    logger.info(f"Training samples: {len(video_paths)}")
    logger.info(f"Sequence length: {args.seq_length}, Batch size: {args.batch_size}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        engagement_loss_total = 0
        confidence_loss_total = 0
        attention_loss_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            frames = batch['frames'].to(device)
            engagement_labels = batch['engagement_label'].to(device)
            confidence_labels = batch['confidence_label'].to(device)
            attention_labels = batch['attention_label'].to(device)
            
            outputs = model(frames)
            
            engagement_loss = criterion(outputs['engagement'].squeeze(), engagement_labels)
            confidence_loss = criterion(outputs['confidence'].squeeze(), confidence_labels)
            attention_loss = criterion(outputs['attention'].squeeze(), attention_labels)
            
            loss = engagement_loss + confidence_loss + attention_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            engagement_loss_total += engagement_loss.item()
            confidence_loss_total += confidence_loss.item()
            attention_loss_total += attention_loss.item()
            
            if batch_idx % 2 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f} '
                   f'(Engagement: {engagement_loss_total/len(train_loader):.4f}, '
                   f'Confidence: {confidence_loss_total/len(train_loader):.4f}, '
                   f'Attention: {attention_loss_total/len(train_loader):.4f})')
    
    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save_model(args.save_path)
    logger.info(f"Training completed. Model saved to {args.save_path}")

if __name__ == '__main__':
    train_vision_model()