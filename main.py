# PyTorch Spam-Ham Classifier (LSTM)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import tokenize, build_vocab, TextDataset
from model import SpamHamLSTM
import torch.nn as nn
import torch.optim as optim
import csv

# 1. Load Data

# Try to load as CSV first, fallback to tab-separated if error
try:
    data = pd.read_csv('spamhamdata.csv')
    texts = data['text'].tolist()
    labels = [1 if l == 'spam' else 0 for l in data['label']]
except Exception:
    # Fallback for tab-separated format (no header)
    texts, labels = [], []
    with open('spamhamdata.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                labels.append(1 if row[0].strip().lower() == 'spam' else 0)
                texts.append(row[1].strip())

vocab = build_vocab(texts)
max_len = 50

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = TextDataset(X_train, y_train, vocab, max_len)
test_dataset = TextDataset(X_test, y_test, vocab, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. Training and Evaluation

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 3. Main

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpamHamLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Acc={acc:.4f}')
    torch.save(model.state_dict(), 'spamham_lstm.pt')
    print('Model saved to spamham_lstm.pt')

if __name__ == '__main__':
    main()
