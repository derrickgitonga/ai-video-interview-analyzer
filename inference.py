import torch
import numpy as np
from models.nlp_model import NLPInterviewAnalyzer
from models.vision_model import VisionInterviewAnalyzer
from utils.text_preprocessing import TextPreprocessor
from utils.video_preprocessing import VideoPreprocessor
from utils.logger import setup_logging
import logging
import json
import os

setup_logging()
logger = logging.getLogger(__name__)

class InterviewAnalyzer:
    def __init__(self, nlp_model_path='models/nlp_model.pth', 
                 vision_model_path='models/vision_model.pth'):
        # Initialize with simple analysis first (bypass model loading)
        self.text_preprocessor = TextPreprocessor()
        self.video_preprocessor = VideoPreprocessor()
        
        # Try to load models, but fallback to simple analysis if they don't exist
        self.nlp_model = None
        self.vision_model = None
        
        try:
            if os.path.exists(nlp_model_path):
                self.nlp_model = NLPInterviewAnalyzer.load_model(nlp_model_path)
                self.nlp_model.eval()
                logger.info("NLP model loaded successfully")
            else:
                logger.warning("NLP model not found. Using simple text analysis.")
                
            if os.path.exists(vision_model_path):
                self.vision_model = VisionInterviewAnalyzer.load_model(vision_model_path)
                self.vision_model.eval()
                logger.info("Vision model loaded successfully")
            else:
                logger.warning("Vision model not found. Using simple video analysis.")
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}. Using simple analysis.")
        
        logger.info("Interview analyzer initialized")

    def analyze_text(self, text, job_description=None):
        """Analyze interview text response - with fallback to simple analysis"""
        if self.nlp_model is None:
            return self._simple_text_analysis(text, job_description)
            
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            with torch.no_grad():
                outputs = self.nlp_model(encoding['input_ids'], encoding['attention_mask'])
            
            # Extract keywords
            keywords = self.text_preprocessor.extract_keywords(text, job_description)
            clarity_metrics = self.text_preprocessor.calculate_clarity_metrics(text)
            
            return {
                'clarity_score': outputs['clarity'].item(),
                'sentiment': torch.argmax(outputs['sentiment']).item(),
                'keyword_matches': keywords,
                'keyword_score': outputs['keywords'].mean().item(),
                'clarity_metrics': clarity_metrics
            }
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}. Falling back to simple analysis.")
            return self._simple_text_analysis(text, job_description)

    def _simple_text_analysis(self, text, job_description=None):
        """Simple text analysis fallback"""
        try:
            cleaned_text = self.text_preprocessor.clean_text(text)
            keywords = self.text_preprocessor.extract_keywords(text, job_description)
            clarity_metrics = self.text_preprocessor.calculate_clarity_metrics(text)
            
            # Simple scoring logic
            word_count = clarity_metrics.get('word_count', 0)
            clarity_score = min(1.0, word_count / 200)  # More words = better clarity
            
            # Simple sentiment analysis
            positive_words = {'good', 'great', 'excellent', 'love', 'enjoy', 'passionate', 'excited'}
            negative_words = {'bad', 'poor', 'hate', 'difficult', 'hard', 'problem', 'challenge'}
            
            words = cleaned_text.split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                sentiment = 2  # positive
            elif negative_count > positive_count:
                sentiment = 0  # negative
            else:
                sentiment = 1  # neutral
                
            keyword_score = min(1.0, len(keywords) / 5)  # More keywords = better match
            
            return {
                'clarity_score': clarity_score,
                'sentiment': sentiment,
                'keyword_matches': keywords,
                'keyword_score': keyword_score,
                'clarity_metrics': clarity_metrics
            }
            
        except Exception as e:
            logger.error(f"Simple text analysis also failed: {e}")
            return {
                'clarity_score': 0.5,
                'sentiment': 1,
                'keyword_matches': [],
                'keyword_score': 0.3,
                'clarity_metrics': {'word_count': 0, 'sentence_count': 0}
            }

    def analyze_video(self, video_path):
        """Analyze interview video - with fallback to simple analysis"""
        if self.vision_model is None:
            return self._simple_video_analysis(video_path)
            
        try:
            frames = self.video_preprocessor.preprocess_video(video_path)
            if frames is None:
                return self._simple_video_analysis(video_path)
                
            with torch.no_grad():
                outputs = self.vision_model(frames)
            
            facial_analysis = self.video_preprocessor.analyze_facial_expressions(video_path)
            
            return {
                'engagement_score': outputs['engagement'].item(),
                'confidence_score': outputs['confidence'].item(),
                'attention_score': outputs['attention'].item(),
                'facial_analysis': facial_analysis
            }
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}. Falling back to simple analysis.")
            return self._simple_video_analysis(video_path)

    def _simple_video_analysis(self, video_path):
        """Simple video analysis fallback"""
        try:
            facial_analysis = self.video_preprocessor.analyze_facial_expressions(video_path)
            
            # Simple scoring based on facial analysis
            engagement = facial_analysis.get('smile_frequency', 0.5) * 0.6 + facial_analysis.get('eye_contact_estimate', 0.5) * 0.4
            confidence = facial_analysis.get('smile_frequency', 0.5) * 0.5 + (1 - facial_analysis.get('head_movement', 0.3)) * 0.5
            attention = facial_analysis.get('eye_contact_estimate', 0.5) * 0.7 + min(1.0, facial_analysis.get('frames_analyzed', 10) / 20) * 0.3
            
            return {
                'engagement_score': min(1.0, engagement),
                'confidence_score': min(1.0, confidence),
                'attention_score': min(1.0, attention),
                'facial_analysis': facial_analysis
            }
            
        except Exception as e:
            logger.error(f"Simple video analysis also failed: {e}")
            return {
                'engagement_score': 0.6,
                'confidence_score': 0.5,
                'attention_score': 0.7,
                'facial_analysis': {}
            }

    def analyze_candidate(self, video_path, transcript, job_description=None, candidate_name="Candidate"):
        """Complete analysis of candidate interview with robust error handling"""
        logger.info(f"Analyzing candidate: {candidate_name}")
        
        try:
            # Analyze text
            text_analysis = self.analyze_text(transcript, job_description)
            
            # Analyze video
            video_analysis = self.analyze_video(video_path)
            
            # Calculate overall score (weighted average)
            overall_score = (
                text_analysis['clarity_score'] * 0.3 +
                text_analysis['keyword_score'] * 0.3 +
                video_analysis['engagement_score'] * 0.2 +
                video_analysis['confidence_score'] * 0.2
            )
            
            results = {
                'candidate_name': candidate_name,
                'overall_score': overall_score,
                'text_analysis': text_analysis,
                'video_analysis': video_analysis,
                'recommendation': self._get_recommendation(overall_score)
            }
            
            logger.info(f"Analysis complete for {candidate_name}. Score: {overall_score:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            # Return a default result instead of crashing
            return {
                'candidate_name': candidate_name,
                'overall_score': 0.5,
                'text_analysis': {
                    'clarity_score': 0.5,
                    'sentiment': 1,
                    'keyword_matches': [],
                    'keyword_score': 0.3,
                    'clarity_metrics': {'word_count': 0, 'sentence_count': 0}
                },
                'video_analysis': {
                    'engagement_score': 0.6,
                    'confidence_score': 0.5,
                    'attention_score': 0.7,
                    'facial_analysis': {}
                },
                'recommendation': 'Consider',
                'error': str(e)
            }

    def _get_recommendation(self, score):
        if score >= 0.8:
            return "Strongly Recommend"
        elif score >= 0.6:
            return "Recommend"
        elif score >= 0.4:
            return "Consider"
        else:
            return "Not Recommended"

    def analyze_multiple_candidates(self, candidates_data, job_description=None):
        """Analyze multiple candidates and return ranked results with error handling"""
        results = []
        
        for candidate in candidates_data:
            try:
                result = self.analyze_candidate(
                    candidate.get('video_path', ''),
                    candidate.get('transcript', ''),
                    job_description,
                    candidate.get('name', 'Unknown Candidate')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze candidate {candidate.get('name', 'unknown')}: {e}")
                # Skip failed candidates instead of crashing
        
        # Sort by overall score descending
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results

def main():
    # Example usage
    analyzer = InterviewAnalyzer()
    
    # Sample data
    candidates = [
        {
            'name': 'John Doe',
            'video_path': 'data/sample_video_1.mp4',
            'transcript': 'I have extensive experience in machine learning and deep learning...'
        }
    ]
    
    with open('data/sample_job_description.txt', 'r') as f:
        job_description = f.read()
    
    results = analyzer.analyze_multiple_candidates(candidates, job_description)
    
    # Save results
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis completed. Results saved to analysis_results.json")

if __name__ == '__main__':
    main()