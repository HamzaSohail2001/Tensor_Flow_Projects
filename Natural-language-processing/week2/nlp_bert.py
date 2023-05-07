import tensorflow as tf
import numpy as np

# Prepare the dataset
data = [
    ("This is a great product", 1),
    ("I'm very satisfied with this purchase", 1),
    ("I hate this product", 0),
    ("This product is terrible", 0),
    ("This product is okay", 2),
    ("It could be better", 2),
]

# Split the data into training and testing sets
train_data = data[:4]
test_data = data[4:]

# Create a vocabulary of words
vocab = set()
for text, label in data:
    for word in text.split():
        vocab.add(word)

# Create a mapping of words to indices
word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

# Convert the data into a format that can be used by the model
def convert_data(data):
    inputs = []
    labels = []
    for text, label in data:
        inputs.append([word2idx[word] for word in text.split()])
        labels.append(label)
    return inputs, labels

train_inputs, train_labels = convert_data(train_data)
test_inputs, test_labels = convert_data(test_data)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(np.array(train_inputs), np.array(train_labels), epochs=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(np.array(test_inputs), np.array(test_labels), verbose=2)
print('\nTest accuracy:', test_acc)
