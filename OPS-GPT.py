# from tensorflow import keras
# import tensorflow as tf
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences

# # Define your training data
# texts = ['This is a positive sentence', 'This is a negative sentence', 'Another positive sentence', 'Another negative sentence']
# labels = [1, 0, 1, 0]  # 1 represents positive and 0 represents negative

# # Tokenize the text
# tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
# tokenizer.fit_on_texts(texts)
# word_index = tokenizer.word_index

# # Convert the text into sequences
# sequences = tokenizer.texts_to_sequences(texts)

# # Pad the sequences
# padded_sequences = pad_sequences(sequences, padding='post')

# # Define the model architecture
# model = keras.Sequential([
#     tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# model.fit(padded_sequences, labels, epochs=10)

from transformers import AutoTokenizer, AutoModel
import torch

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Input sentences to compare
sentence1 = "The cat is sleeping on the mat."
sentence2 = "The dog is sleeping on the rug."

# Tokenize the input sentences
inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', 
                               max_length=128, truncation=True)

# Get the BERT model outputs for the input sentences
outputs = model(**inputs)

# Extract the output representations for the [CLS] token for each sentence
sentence1_rep = outputs.last_hidden_state[0][0]
sentence2_rep = outputs.last_hidden_state[0][1]

# Compute the cosine similarity between the sentence representations
cosine_sim = torch.nn.functional.cosine_similarity(sentence1_rep, sentence2_rep, dim=0)

print("Sentence 1: ", sentence1)
print("Sentence 2: ", sentence2)
print("Similarity score: ", cosine_sim.item())
