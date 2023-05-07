import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


sentences = [
    "I love my dog",
    "I love my cat",
    "I admire momina",
    "Momina has a melodious voice "
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index


test = [
    "I love your melodious voice,  Momina ",
    " I love your dog biscuit as well"
] 
sequence = tokenizer.texts_to_sequences(test) 
sequence_test = tokenizer.texts_to_sequences(test)
print(word_index)
print(sequence_test)
padded = pad_sequences(sequence_test,padding='post',truncating='post',maxlen=10) 
print(padded)