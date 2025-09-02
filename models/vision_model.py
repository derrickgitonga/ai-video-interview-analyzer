import torch
import torch.nn as nn
import torchvision.models as models
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VisionInterviewAnalyzer(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.4):
        super(VisionInterviewAnalyzer, self).__init__()
        
        # Use OpenCV's Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Main feature extractor - ResNet50
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Additional feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Calculate the total input dimension for classifiers
        # ResNet features (hidden_dim) + visual features (2) = hidden_dim + 2
        classifier_input_dim = hidden_dim + 2
        
        # Classifiers
        self.engagement_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.attention_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Vision model initialized. Classifier input dim: {classifier_input_dim}")

    def detect_faces(self, frame):
        """Detect faces in a frame using OpenCV"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            return len(faces) > 0, faces
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return False, []

    def extract_visual_features(self, frame, faces):
        """Extract visual features from frame and face information"""
        try:
            # Basic face presence feature
            face_present = 1.0 if len(faces) > 0 else 0.0
            
            # Simple face size feature (normalized)
            face_size = 0.0
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                face_size = min(1.0, (largest_face[2] * largest_face[3]) / (frame.shape[0] * frame.shape[1] * 0.1))
            
            # Convert to tensor and return basic features
            return torch.tensor([face_present, face_size], dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            return torch.tensor([0.0, 0.0], dtype=torch.float32)

    def forward(self, x):
        # x is batch of video frames: [batch_size, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = x.shape
        
        # Process each frame through ResNet
        x_flat = x.view(batch_size * seq_len, C, H, W)
        resnet_features = self.resnet(x_flat)
        processed_features = self.feature_processor(resnet_features)
        
        # Reshape back to [batch_size, seq_len, hidden_dim]
        processed_features = processed_features.view(batch_size, seq_len, -1)
        
        # Extract frame-level visual features
        visual_features_list = []
        for i in range(batch_size):
            for j in range(seq_len):
                frame = (x[i, j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                has_face, faces = self.detect_faces(frame)
                visual_feat = self.extract_visual_features(frame, faces)
                visual_features_list.append(visual_feat)
        
        visual_features = torch.stack(visual_features_list).view(batch_size, seq_len, -1)
        
        # Average features over time sequence
        avg_processed_features = processed_features.mean(dim=1)
        avg_visual_features = visual_features.mean(dim=1)
        
        # Combine ResNet features with visual features
        combined_features = torch.cat([avg_processed_features, avg_visual_features], dim=1)
        
        # Get predictions
        engagement = self.engagement_classifier(combined_features)
        confidence = self.confidence_classifier(combined_features)
        attention = self.attention_classifier(combined_features)
        
        return {
            'engagement': engagement,
            'confidence': confidence,
            'attention': attention
        }

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.feature_processor[0].out_features,
                'dropout': self.feature_processor[2].p
            }
        }, path)
        logger.info(f"Vision model saved to {path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Vision model loaded from {path}")
        return model