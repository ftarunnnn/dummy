import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json

class TextSentimentNet(nn.Module):
    """
    Bi-directional LSTM / Attention Neural Network for NLP sentiment classification
    and text analysis.
    """
    SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=64, num_classes=3):
        super(TextSentimentNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        # Max pooling across time dimension
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        out = self.fc2(out)
        return out

    def build_vocab(self, texts, max_words=5000):
        """Construct vocabulary mapping from input text corpus."""
        word_freq = {}
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words - 2]
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.vocab[word] = idx

    def _tokenize(self, text):
        """Clean and tokenize text."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def text_to_tensor(self, text, max_len=30):
        """Convert raw string to token ID tensor."""
        tokens = self._tokenize(text)
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if len(ids) < max_len:
            ids = ids + [self.vocab["<PAD>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor([ids], dtype=torch.long)

    def predict_sentiment(self, text):
        """Predict sentiment label and probabilities for raw text input."""
        self.eval()
        with torch.no_grad():
            tensor = self.text_to_tensor(text)
            logits = self.forward(tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())
        
        return {
            "text": text,
            "sentiment": self.SENTIMENT_LABELS[pred_idx],
            "confidence": float(round(probs[pred_idx] * 100, 2)),
            "probabilities": {
                label: float(round(prob * 100, 2))
                for label, prob in zip(self.SENTIMENT_LABELS, probs)
            }
        }
