import pandas as pd
import gensim
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import numpy as np
import re

# Load data
df = pd.read_csv('Data/spamhamdata.csv', sep='\t', header=None, names=['class', 'text'])



# Label encoding
le = LabelEncoder()
y = le.fit_transform(df['class'])


sentences = df['text']
lemmatizer = WordNetLemmatizer()
def clean_and_lemmatize(text):
	review = re.sub('[^a-zA-Z]', ' ', text)
	review = review.lower().split()
	return [lemmatizer.lemmatize(word) for word in review]
corpus = [clean_and_lemmatize(sentence) for sentence in sentences]

vocab_size = 3570
max_input_length = 190


# Tokenizer: save/load to avoid refitting every run
import os, pickle
tokenizer_path = 'RNN_Model/tokenizer.pkl'
if os.path.exists(tokenizer_path):
	with open(tokenizer_path, 'rb') as f:
		tokenizer = pickle.load(f)
else:
	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(corpus)
	with open(tokenizer_path, 'wb') as f:
		pickle.dump(tokenizer, f)

# Load trained model
model = load_model('RNN_Model/Spam_Classifier.h5')

def preprocess_message(message):
	review = re.sub('[^a-zA-Z]', ' ', message)
	review = review.lower()
	review = review.split()
	review = [lemmatizer.lemmatize(word) for word in review]
	return review

def predict_class(text):
	processed_text = preprocess_message(text)
	sequence = tokenizer.texts_to_sequences([processed_text])
	padded_sequence = pad_sequences(sequence, maxlen=max_input_length)
	prediction = model.predict(padded_sequence)
	predicted_class = le.inverse_transform(prediction.argmax(axis=1))[0]
	return predicted_class

if __name__ == '__main__':
	test_messages = [
		'Congratulations! You have won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.',
		'Hey, are we still meeting for lunch today?',
		'URGENT! Your account has been compromised. Reply with your password to secure it.',
		'Can you send me the notes from yesterday’s class?'
	]
	for msg in test_messages:
		result = predict_class(msg)
		print(f'Message: "{msg}"\nPrediction: {result}\n')
