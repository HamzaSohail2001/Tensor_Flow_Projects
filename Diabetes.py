import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split



df = pd.read_csv('diabetes.csv')
print(df.head())
for index in range(len(df.columns)):
    label = df.columns[index]
    plt.hist(df[df['Outcome']==1][label],color='blue')
    plt.hist(df[df['Outcome']==0][label],color='red') 
    plt.title(label)
    plt.show()



X= df[df.columns[:-1]].values

y = df[df.columns[-1]].values
print(X.shape)

print(y.shape)
print(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)



model = tf.keras.Sequential([

    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),

])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=16, epochs=50)


array = [[0,70,100,15,20,23.6,0.627,50]]
argv = model.predict(array)
print(argv)

def prediction(arg):
    if arg > 0.5:
        print("You have diabetes !")
    else:
        print("You do not have diabetes !")
prediction(argv)
