import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import *

class dataset:
    def __init__(self, path, name, threshold):
        self.df = get_df_datas(path, name, threshold)
        self.datas = list(range(len(self.df)-50))
        self.labels = list(np.ceil(self.df['CO(GT)'][50:]))

    def split(self, s):
        X_train, X_test, y_train, y_test = train_test_split(self.datas, self.labels, test_size=s, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_len = len(X_train)
        self.test_len = len(X_test)

    def generate(self):
        def ds_generator():
            while True:
                for i in range(self.train_len):
                    x = np.array(self.df.iloc[self.X_train[i]:self.X_train[i]+50])
                    y = self.y_train[i]
                    yield (x,y)
        def vali_generator():
            while True:
                for i in range(self.test_len):
                    x = np.array(self.df.iloc[self.X_test[i]:self.X_test[i]+50])
                    y = self.y_test[i]
                    yield (x,y)

        dataset = tf.data.Dataset.from_generator(
            ds_generator,
            output_signature=(
                tf.TensorSpec(shape=(50,17), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        validation = tf.data.Dataset.from_generator(
            vali_generator,
            output_signature=(
                tf.TensorSpec(shape=(50,17), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        dataset = dataset.shuffle(buffer_size=100).batch(32)
        validation = validation.batch(32)

        self.dataset = dataset
        self.validation = validation





















        
        