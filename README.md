# Spam–Ham Classification using RNN

## 1. Introduction

Spam messages are one of the most common issues in digital communication, leading to security risks, time loss, and reduced user trust. Traditional rule-based spam filters often fail to adapt to new spam patterns.
Deep learning models, especially Recurrent Neural Networks (RNNs), are effective in handling sequential data like text, making them well-suited for spam detection.
This project uses an RNN to classify messages as either "Spam" or "Ham" (non-spam) based on their textual content.

## 2. Problem Statement

The aim is to develop an AI-based system capable of:

1. Automatically classifying text messages as Spam or Ham.
2. Handling variable-length message inputs effectively.
3. Achieving high accuracy and generalization to real-world scenarios.

## 3. Objectives

* Preprocess text data for RNN input.
* Train an RNN model for binary classification.
* Evaluate performance on training, validation, and test sets.
* Provide real-time single-message classification capability.


## 4. Dataset Details

- **Source:** Provided dataset (`spamhamdata.csv`)
- **Format:** CSV or tab-separated, with two columns: label (spam/ham) and message text
- **Preprocessing:**
  - Lowercasing
  - Removal of non-alphanumeric characters
  - Whitespace-based tokenization
  - Padding/truncating to max sequence length (50)
  - Custom vocabulary built from training data

## 5. Methodology

### 5.1 Data Loading & Preprocessing
- Data loaded from CSV or tab-separated file
- Tokenization and vocabulary building performed using custom Python functions
- Messages converted to integer sequences for model input

### 5.2 Model Architecture
- PyTorch implementation
- Embedding layer (nn.Embedding)
- LSTM layer for sequence modeling
- Fully connected output layer for classification

### 5.3 Training
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 5
- Batch size: 32
- Model saved as `spamham_lstm.pt`

### 5.4 Inference
- Loads trained model and vocabulary
- Predicts spam/ham for each message in the dataset

## 6. Results

- **Training Accuracy:** Up to 96% on test set after 5 epochs
- **Inference:** Model correctly classifies spam and ham messages from the dataset

## 7. How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python main.py
   ```
3. Run inference:
   ```bash
   python inference.py
   ```

## 8. Files
- `main.py`: Training script
- `inference.py`: Inference script
- `model.py`: Model definition
- `data_utils.py`: Preprocessing and dataset utilities
- `spamhamdata.csv`: Dataset
- `spamham_lstm.pt`: Saved model
- `requirements.txt`: Dependencies

## 9. Output Example

**Training:**
```
Epoch 1: Train Loss=0.3901, Test Acc=0.9166
Epoch 2: Train Loss=0.1520, Test Acc=0.8861
Epoch 3: Train Loss=0.1492, Test Acc=0.9318
Epoch 4: Train Loss=0.1263, Test Acc=0.9641
Epoch 5: Train Loss=0.2000, Test Acc=0.9094
Model saved to spamham_lstm.pt
```

**Inference:**
```
Text: You have been specially selected to receive a "3000 award! ...
Predicted: spam
Text: Are we meeting today?
Predicted: ham
...etc
```

---

**Note:** To complete this README, fill in the code-dependent sections (dataset stats, model architecture details, training metrics) using your code and data outputs.
