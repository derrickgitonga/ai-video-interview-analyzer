import re
import logging
import ssl

class TextPreprocessor:
    def __init__(self):
        # Basic English stopwords list (no nltk dependency)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }
        
        # Try to use NLTK if available, but fallback to basic tokenization
        self.use_nltk = False
        try:
            import nltk
            # Try to load punkt tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
                self.use_nltk = True
                from nltk.tokenize import word_tokenize
                self.word_tokenize = word_tokenize
                logging.info("NLTK tokenizer loaded successfully")
            except LookupError:
                logging.warning("NLTK punkt tokenizer not found. Using basic tokenization.")
                self.word_tokenize = self.basic_tokenize
        except ImportError:
            logging.warning("NLTK not available. Using basic tokenization.")
            self.word_tokenize = self.basic_tokenize

    def basic_tokenize(self, text):
        """Basic tokenization without NLTK"""
        return text.split()

    def clean_text(self, text):
        """Clean and preprocess text"""
        try:
            if not isinstance(text, str):
                text = str(text)
                
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = self.word_tokenize(text)
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            logging.error(f"Text cleaning failed: {e}")
            return text

    def extract_keywords(self, text, job_description=None):
        """Extract keywords from text, optionally matching job description"""
        cleaned_text = self.clean_text(text)
        tokens = self.basic_tokenize(cleaned_text)
        
        if job_description and job_description.strip() and job_description != "Enter the job description for keyword matching...":
            cleaned_job_desc = self.clean_text(job_description)
            job_keywords = set(self.basic_tokenize(cleaned_job_desc))
            matched_keywords = [token for token in tokens if token in job_keywords]
            return list(set(matched_keywords))
        
        return list(set(tokens))

    def calculate_clarity_metrics(self, text):
        """Calculate text clarity metrics"""
        if not text or not isinstance(text, str):
            return {
                'avg_sentence_length': 0,
                'unique_word_ratio': 0,
                'word_count': 0,
                'sentence_count': 0
            }
            
        # Use basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'unique_word_ratio': unique_word_ratio,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }