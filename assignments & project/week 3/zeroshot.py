from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

class FinBERTAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Load pre-trained FinBERT."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    def analyze_sentiment_zero_shot(self, text):
        """
        Get sentiment prediction without fine-tuning.
        Returns: {predicted_label, confidence, probabilities}
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            'text': text,
            'predicted_label': self.label_map[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'positive': probabilities[0][0].item(),
                'negative': probabilities[0][1].item(),
                'neutral': probabilities[0][2].item()
            }
        }

    def analyze_batch(self, texts):
        """Efficiently analyze multiple texts."""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment_zero_shot(text))
        return results
