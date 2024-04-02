import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load and preprocess dataset
df = pd.read_csv("sentiment_data.csv")  # Replace "sentiment_data.csv" with your dataset file
sentences = df['text'].values
labels = df['label'].values

# Split dataset into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)

# Tokenize texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_labels))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']),
                             torch.tensor(test_labels))

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: positive, negative, neutral

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(3):  # 3 epochs for demonstration
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    scheduler.step()

# Evaluation
model.eval()
val_preds = []
val_labels = []
for batch in val_loader:
    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    val_preds.extend(torch.argmax(logits, dim=1).tolist())
    val_labels.extend(labels.tolist())

# Calculate accuracy on validation set
val_accuracy = accuracy_score(val_labels, val_preds)
print("Validation Accuracy:", val_accuracy)

# Test
test_preds = []
test_labels = []
for batch in test_loader:
    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    test_preds.extend(torch.argmax(logits, dim=1).tolist())
    test_labels.extend(labels.tolist())

# Calculate accuracy on test set
test_accuracy = accuracy_score(test_labels, test_preds)
print("Test Accuracy:", test_accuracy)
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 classes: positive, negative, neutral

# Load the trained model weights
model.load_state_dict(torch.load("bert_sentiment_model.pth"))  # Replace "bert_sentiment_model.pth" with your trained model file

# Function for real-time prediction
def predict_sentiment(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_sentiment = sentiment_labels[predicted_label]
    return predicted_sentiment

# Example usage
input_text = "I loved the movie, it was fantastic!"
predicted_sentiment = predict_sentiment(input_text)
print("Predicted sentiment:", predicted_sentiment)
