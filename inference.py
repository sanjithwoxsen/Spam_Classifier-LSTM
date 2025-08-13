import pandas as pd
import torch
from data_utils import tokenize, build_vocab, TextDataset
from model import SpamHamLSTM
import csv

# Load the training dataset to build vocab
try:
    data = pd.read_csv('spamhamdata.csv')
    texts = data['text'].tolist()
    labels = [1 if l == 'spam' else 0 for l in data['label']]
except Exception:
    texts, labels = [], []
    with open('spamhamdata.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                labels.append(1 if row[0].strip().lower() == 'spam' else 0)
                texts.append(row[1].strip())

vocab = build_vocab(texts)
max_len = 50

dataset = TextDataset(texts, labels, vocab, max_len)

model = SpamHamLSTM(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_classes=2)
model.load_state_dict(torch.load('spamham_lstm.pt', map_location='cpu'))
model.eval()

with torch.no_grad():
    for i in range(len(dataset)):
        x, _ = dataset[i]
        x = x.unsqueeze(0)
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        label = 'spam' if pred == 1 else 'ham'
        print(f"Text: {texts[i]}\nPredicted: {label}\n")
