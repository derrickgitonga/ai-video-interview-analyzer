import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging

logger = logging.getLogger(__name__)

class NLPInterviewAnalyzer(nn.Module):
    def __init__(self, num_keywords=10, hidden_dim=256, dropout=0.3):
        super(NLPInterviewAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers for fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.clarity_classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # positive, negative, neutral
            nn.Softmax(dim=1)
        )
        
        self.keyword_matcher = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_keywords),
            nn.Sigmoid()
        )
        
        logger.info("NLP model initialized with BERT base")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        clarity_score = self.clarity_classifier(pooled_output)
        sentiment_probs = self.sentiment_classifier(pooled_output)
        keyword_scores = self.keyword_matcher(pooled_output)
        
        return {
            'clarity': clarity_score,
            'sentiment': sentiment_probs,
            'keywords': keyword_scores
        }

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_keywords': self.keyword_matcher[-2].out_features,
                'hidden_dim': self.clarity_classifier[0].out_features,
                'dropout': self.clarity_classifier[2].p
            }
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return model