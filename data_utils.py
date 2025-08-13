import re
from collections import Counter
import torch
from torch.utils.data import Dataset

def tokenize(text):
    # Remove leading label if present (for tab-separated dataset)
    if '\t' in text:
        parts = text.split('\t', 1)
        if len(parts) == 2:
            text = parts[1]
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        seq = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        seq = seq[:self.max_len] + [self.vocab['<PAD>']] * (self.max_len - len(seq))
        return torch.tensor(seq), torch.tensor(self.labels[idx])
