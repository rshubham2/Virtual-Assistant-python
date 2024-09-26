import torch
from transformers import BertTokenizer, BertForSequenceClassification

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


MAX_LENGTH = 128

def load_emotion_model():
    # Define the list of emotions
    emotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
    
    # Load pre-trained BERT model for classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotions))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, emotions

def predict_emotion(model, tokenizer, text, max_length, device, emotions):
    # Tokenize and prepare the input text for the model
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    model = model.to(device)

    # Move input tensors to the same device as the model
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0, predicted_label].item()
    predicted_emotion = emotions[predicted_label]  # Map label to emotion

    return predicted_emotion, confidence