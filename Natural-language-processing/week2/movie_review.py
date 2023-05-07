import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

imdb, info = tfds.load("imdb_reviews",with_info=True, as_supervised=True)
train_data , test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())


for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())


training_sentences_final = np.array(training_labels)
testing_sentences_final = np.array(testing_labels)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=120,truncating='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=120)


model= tf.keras.Sequential([
  tf.keras.layers.Embedding(10000,16,input_length=120),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(6, activation='relu'),
   tf.keras.layers.Dense(1, activation='sigmoid'),
])

# model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
# model.summary()
# model.fit(padded,training_sentences_final,epochs=10,validation_data=(testing_padded,testing_sentences_final))

# model.save('my_model.h5')
# Load the saved model from a file
loaded_model = tf.keras.models.load_model('my_model.h5')

array = ["the movie was a bit boring not worth watching it"]
loaded_model.predict(array)
