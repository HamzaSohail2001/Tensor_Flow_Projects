import io
import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

filename = 'Natural-language-processing/week2/BBC News Train.csv'

def stop_words(sentence):

    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    sentence = sentence.lower()
    words = sentence.split()
    para = ""
    for x in words:
        if x not in stopwords:
            para += x
            para += " "
    return para

def parse(filename):
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[2])
            sentence = row[1]
            sentence = stop_words(sentence)
            sentences.append(sentence)

    return sentences, labels


sentences, labels = parse(filename)

print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}")


# we will now split the data, so we can prepare for training an testing 

def train_val_split(sentences, labels, training_split):

    train_size = int(len(sentences)*training_split)
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentence  = sentences[train_size:]
    validation_labels = labels[train_size:]

    return train_sentences, validation_sentence, train_labels, validation_labels



train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, 0.8)

print(f"There are {len(train_sentences)} sentences for training.\n")
print(f"There are {len(train_labels)} labels for training.\n")
print(f"There are {len(val_sentences)} sentences for validation.\n")
print(f"There are {len(val_labels)} labels for validation.")


def fit_tokenizer(train , words, oov):
    # in this function we will tokenizing the words in the training sequence 
    tokenizer = Tokenizer(num_words=words,oov_token=oov)
    tokenizer.fit_on_texts(train)

    return tokenizer

tokenizer = fit_tokenizer(train_sentences, 1000, "<OOV>")
word_index = tokenizer.word_index

print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")


def seq_and_pad(sentences, tokenizer, padding, maxlen):

       
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Pad the sequences using the correct padding and maxlen
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding)
    
    return padded_sequences

NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8

train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)

print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
print(f"Padded validation sequences have shape: {val_padded_seq.shape}")


def tokenize_labels(all_labels, split_labels):
    
   
    label_tokenizer = Tokenizer()
    
    label_tokenizer.fit_on_texts(all_labels)

    label_seq = label_tokenizer.texts_to_sequences(split_labels)
    
  
    label_seq_np = np.array(label_seq) - 1

     
    return label_seq_np


train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

print(f"First 5 labels of the training set should look like this:\n{train_label_seq[:5]}\n")
print(f"First 5 labels of the validation set should look like this:\n{val_label_seq[:5]}\n")
print(f"Tokenized labels of the training set have shape: {train_label_seq.shape}\n")
print(f"Tokenized labels of the validation set have shape: {val_label_seq.shape}\n")


def create_model(num_words, embedding_dim, maxlen):

    tf.random.set_seed(123)
    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24,activation='relu'),
        tf.keras.layers.Dense(5,activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 
    
    return model

model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)

history = model.fit(train_padded_seq, train_label_seq, epochs=30, validation_data=(val_padded_seq, val_label_seq))




new_sentence = "blair rejects iraq advice calls tony blair has rejected calls for the publication of advice on the legality of the iraq war amid growing calls for an investigation.  the prime minister told his monthly press conference the matter had been dealt with by the attorney general. earlier  conservative mp michael mates joined calls for a probe into claims lord goldsmith s statement to parliament was drawn up at number 10. mr blair said the statement was a  fair summary  of lord goldsmith s opinion.   that s what he (lord goldsmith) said and that s what i say. he has dealt with this time and time and time again   mr blair told his monthly news conference in downing street. he refused to answer further questions on the issue  saying it had been dealt with  literally scores of times and the position has not changed . lord goldsmith has denied being  leaned on  and says the words written were his.  the government refuses to publish his advice on the legality of the war - saying such papers have always been kept confidential. mr mates  who is a member of the commons intelligence and security committee and was part of the butler inquiry into pre-war intelligence  told the bbc on friday:  that  as a general rule  is right  but it s not an absolute rule.  he said there had been other occasions when advice had been published  most recently regarding prince charles s marriage plans. the government could not pick and choose when to use the convention  he said.  mr mates added:  we discovered that there were two or three occasions in the past when law officers  advice to the government has been published.  and this may be one of those special occasions... when it would be in the public interest to see the advice which the attorney general gave to the prime minister.  this is argument was rejected by mr blair  who said:  firstly  we haven t broken the precedent  and secondly peter goldsmith has made his statement and i have got absolutely nothing to add to it.  in a book published this week  philippe sands qc  a member of cherie blair s matrix chambers  says lord goldsmith warned tony blair on 7 march 2003 that the iraq war could be illegal without a second un resolution sanctioning military action.  but a short statement about lord goldsmith s position was presented in a written parliamentary answer on 17 march 2003 - just before a crucial commons vote on the military action. mr sands  book suggests it was actually written by home office minister lord falconer and downing street adviser baroness morgan. former minister clare short  who resigned from the government over the iraq war  said it was the same statement that was earlier shown to the cabinet as it discussed military action. she told the bbc the full advice should have been attached  according to the ministerial code.   my view is we need the house of lords to set up a special committee  summon the attorney  get all the papers out  look at exactly what happened   she said. the conservatives and liberal democrats say they want the publication of the full legal advice given by the attorney general. on thursday  lord goldsmith said his statement had not been  written by or at number 10 .  in my parliamentary answer on march 17 2003  i explained my genuinely held independent view  that military action was lawful under the existing security council resolutions   he said."
new_sequence = tokenizer.texts_to_sequences([new_sentence])
new_padded_sequence = pad_sequences(new_sequence, maxlen=MAXLEN, padding=PADDING)

# Obtain predicted class probabilities for the new sentence
class_probabilities = model.predict(new_padded_sequence)

# Print the predicted class probabilities
print(class_probabilities)