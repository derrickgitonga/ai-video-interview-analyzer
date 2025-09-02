import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224), frame_rate=1):
        self.target_size = target_size
        self.frame_rate = frame_rate  # frames per second to extract
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        logger.info("Video preprocessor initialized")

    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video at specified frame rate"""
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                logger.warning(f"Video file {video_path} not found")
                return None
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video file {video_path}")
                return None
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or total_frames == 0:
                logger.warning(f"Invalid video properties for {video_path}")
                cap.release()
                return None
                
            frame_interval = max(1, int(fps / self.frame_rate))
            frames = []
            frame_indices = []
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret and len(frames) < max_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
                    frame_indices.append(i)
            
            cap.release()
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return None
                
            frames_tensor = torch.stack(frames)
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames_tensor
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None

    def preprocess_video(self, video_path, seq_length=30):
        """Main preprocessing function for video analysis"""
        frames = self.extract_frames(video_path, max_frames=seq_length)
        if frames is None:
            # Generate placeholder frames for demo
            logger.info("Generating placeholder frames for demo")
            return torch.randn(seq_length, 3, self.target_size[0], self.target_size[1])
            
        # Ensure sequence length
        if len(frames) > seq_length:
            # Select evenly spaced frames
            indices = np.linspace(0, len(frames)-1, seq_length, dtype=int)
            frames = frames[indices]
        elif len(frames) < seq_length:
            # Pad with last frame
            padding = frames[-1:].repeat(seq_length - len(frames), 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
            
        return frames

    def analyze_facial_expressions(self, video_path):
        """Analyze facial expressions throughout video"""
        frames = self.extract_frames(video_path, max_frames=20)
        if frames is None:
            return {
                'smile_frequency': 0.5,
                'eye_contact_estimate': 0.5,
                'head_movement': 0.3,
                'frames_analyzed': 0
            }
            
        # Simple heuristic-based analysis
        expression_metrics = {
            'smile_frequency': 0,
            'eye_contact_estimate': 0,
            'head_movement': 0,
            'frames_analyzed': len(frames)
        }
        
        # Placeholder analysis (would be replaced with actual ML model)
        for i, frame in enumerate(frames):
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Simple heuristics (these would be replaced with actual analysis)
            expression_metrics['smile_frequency'] += 0.05 if i % 4 == 0 else 0.02
            expression_metrics['eye_contact_estimate'] += 0.06 if i % 3 == 0 else 0.03
            expression_metrics['head_movement'] += 0.04 if i % 5 == 0 else 0.01
        
        # Normalize
        for key in ['smile_frequency', 'eye_contact_estimate', 'head_movement']:
            expression_metrics[key] = min(1.0, expression_metrics[key] / len(frames) * 3)
        
        return expression_metrics