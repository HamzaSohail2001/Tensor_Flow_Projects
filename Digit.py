import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras




Sample_dataset = tfds.load("mnist", split="train", try_gcs=True)
print(Sample_dataset)
