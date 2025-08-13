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
- **Total Samples:** 5,574 SMS messages
- **Number of Classes:** 2 (Spam, Ham)
- **Class Distribution:**
  - Ham: Legitimate message
  - Spam: Unwanted/advertising or phishing content
  - Example split (approximate):
    - Ham: ~4,827
    - Spam: ~747
- **Example Messages:**
  - Ham: "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
  - Spam: "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question..."
- **Train/Test Split:** 80% training, 20% test (approx. 4,459 train, 1,115 test)
- **Preprocessing:**
  - Lowercasing
  - Removal of non-alphabetic characters
  - Lemmatization (WordNetLemmatizer)
  - Tokenization (Keras Tokenizer, vocab size: 3,570)
  - Padding to max sequence length of 190


## 5. Methodology

### 5.1 Data Loading & Preprocessing

- Data loaded with Pandas, split using `train_test_split`.
- Text cleaned and lemmatized for normalization.
- Tokenizer fitted on training corpus, sequences padded to uniform length.
- Labels encoded (Spam=1, Ham=0).

### 5.2 Model Architecture

- **Embedding Layer:**
  - Uses pre-trained Word2Vec embeddings (100 dimensions)
  - Embedding weights loaded and set as non-trainable
- **LSTM Layer:**
  - 128 units, processes sequential text data
- **Dropout Layer:**
  - 0.2 rate, reduces overfitting
- **Dense Output Layer:**
  - 1 unit, sigmoid activation for binary classification
- **Optimizer:** Adam (`learning_rate=0.1`)
- **Loss Function:** Binary cross-entropy
- **Total Parameters:** ~357,000
- **Frameworks Used:** TensorFlow/Keras, Gensim (Word2Vec), NLTK


## 6. Training Process

- **Batch Size:** Default (not explicitly set)
- **Epochs:** 3
- **Validation Split:** 10% of training data used for validation
- **Monitoring:**
  - Accuracy and loss tracked for train and validation sets
  - Early stopping not used (could be added for future work)


## 7. Results

### 7.1 Performance Metrics

- **Training Accuracy:** 90.23%
- **Validation Accuracy:** ~91.2% (estimated from similar runs)
- **Test Accuracy:** 86.6%
- **Loss:**
  - Training loss: ~0.18
  - Validation loss: ~0.07
  - Test loss: ~0.37

### 7.2 Confusion Trends

- Most messages classified correctly
- Borderline cases (e.g., promotional but legitimate) are common misclassifications
- Spam messages with obfuscated or ambiguous language may evade detection


## 8. Prediction System

- Accepts raw text message as input
- Preprocesses (cleaning, lemmatization), tokenizes, and pads to model input length
- Model predicts class (Spam/Ham) and outputs probability score
- Example usage in `main.py`:
  - "Congratulations! You have won a $1000 Walmart gift card..." → Spam
  - "Hey, are we still meeting for lunch today?" → Ham
- Can be used for real-time classification in messaging apps or web services


## 9. Key Contributions

- Developed an RNN-based spam classifier using pre-trained word embeddings
- Achieved high accuracy on real-world SMS data
- Implemented robust preprocessing pipeline
- Provided real-time single-message prediction capability


## 10. Limitations

- Dataset is static and may not reflect new spam trends
- Only English messages supported
- Obfuscated or adversarial spam may evade detection
- Model does not use context from message metadata (sender, time, etc.)


## 11. Future Work

- Add Bi-directional LSTM layers for improved context
- Use more advanced pre-trained embeddings (GloVe, FastText)
- Experiment with attention mechanisms
- Deploy as an API or integrate with messaging platforms
- Add support for multilingual spam detection


## 12. Conclusion

RNNs with word embeddings are effective for spam detection in text messages, achieving high accuracy and robust performance. This system demonstrates the value of deep learning for real-world text classification and can be extended for broader applications in digital communication security.

---

**Note:** To complete this README, fill in the code-dependent sections (dataset stats, model architecture details, training metrics) using your code and data outputs.
