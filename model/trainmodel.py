import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
from emotion_model import EmotionDataset, load_emotion_model, predict_emotion


def load_data():
    data = pd.read_csv("emotion_sample_500.csv")
    texts = data['situation'].tolist()
    labels = data['emotion'].tolist()

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    emotions = label_encoder.classes_.tolist()

    # Check that emotions are correctly defined here
    print(f"Emotions extracted: {emotions}")

    return train_test_split(texts, numeric_labels, test_size=0.2, random_state=42), emotions, label_encoder

def train_model(model, train_loader, val_loader, device, emotions, epochs=5, save_path="best_emotion_model.pth"):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path
            torch.save(model.state_dict(), best_model_path)

    # Save the path of the best model
    joblib.dump({'best_model_path': best_model_path}, "model_info.pkl")
    print(f"Best model saved at: {best_model_path}")

def main():
    # Load data and model
    (train_texts, val_texts, train_labels, val_labels), emotions, label_encoder = load_data()
    print(f"Emotions in the dataset: {emotions}")
    
    # Load model and tokenizer
    emotion_model, tokenizer = load_emotion_model(num_labels=len(emotions))
    
    # Prepare datasets and dataloaders
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emotion_model.to(device)
    
    # Train the model
    train_model(emotion_model, train_loader, val_loader, device, emotions)
    
    # Load the trained model for inference
    emotion_model.load_state_dict(torch.load("best_emotion_model.pth"))
    
    # User input loop
    print("\nNow you can test the model with your own inputs.")
    print("Type 'quit' to exit the program.")
    while True:
        user_input = input("\nEnter a text to analyze: ")
        if user_input.lower() == 'quit':
            break
        # Pass all required arguments to predict_emotion
        emotion, confidence = predict_emotion(emotion_model, tokenizer, user_input, max_length, device, emotions)
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()