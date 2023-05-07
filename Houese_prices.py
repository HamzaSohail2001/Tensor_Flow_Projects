import tensorflow as tf
from tensorflow import keras
import numpy as np
model = keras.Sequential([keras.layers.Dense(units =1 , input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
# dense means layer in the neural layer, units = 1 ,means a single layer
# convergence is used when 100% accuracy is reached 
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0,5.0,7.0],dtype=float)
model.fit(xs,ys,epochs=500) # train the tensorflow model
result = model.predict([10,12 ,14])
print(result)
print(tf.__version__)
