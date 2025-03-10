# components/ai_model.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from components.model_downloader import download_model

class SentimentAnalyzer:
    def __init__(self, model_config):
        """Initialize the sentiment analyzer."""
        self.logger = logging.getLogger('SentimentAnalyzer')
        self.model_name = model_config['sentiment_analysis']['model_name']
        self.use_gpu = model_config['sentiment_analysis']['use_gpu'] and torch.cuda.is_available()
        self.auto_download = model_config['sentiment_analysis']['auto_download']
        
        self.logger.info(f"Loading sentiment model: {self.model_name}")
        try:
            if self.auto_download:
                # Automatically download the model
                model_path = download_model(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                # Use Hugging Face's default downloading mechanism
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.to("cuda")
                self.logger.info("Sentiment model loaded on GPU")
            else:
                self.logger.info("Sentiment model loaded on CPU")
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment of texts using the loaded model."""
        results = []
        
        for text in texts:
            try:
                inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
                
                # Move inputs to GPU if model is on GPU
                if self.use_gpu:
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get prediction
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
                # Map prediction to label (specific to FinBERT)
                labels = ["negative", "neutral", "positive"]
                if len(labels) > prediction:
                    label = labels[prediction]
                else:
                    label = "unknown"
                
                results.append({
                    'label': label,
                    'confidence': confidence
                })
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment: {e}")
                results.append({
                    'label': 'neutral',
                    'confidence': 0.33
                })
        
        return results
