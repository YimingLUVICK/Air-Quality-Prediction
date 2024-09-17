import tensorflow as tf
import pandas as pd

class logcallback(tf.keras.callbacks.Callback):
    def __init__(self, log_path):
        super(logcallback, self).__init__()
        self.log_path = log_path

    def on_train_begin(self, logs=None):
        self.df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        
    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[epoch+1] = [epoch+1, logs.get('loss'), logs.get('accuracy'), logs.get('val_loss'), logs.get('val_accuracy')]

    def on_train_end(self, logs=None):
        self.df.to_csv(self.log_path, index=False)

class trainer:
    def __init__(self, log_path, model_path, ds, model):
        self.logger = logcallback(log_path)
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        self.ds = ds
        self.model = model

    def train(self, e):
        history = self.model.fit(self.ds.dataset, epochs=e, steps_per_epoch=self.ds.train_len//32, validation_data=self.ds.validation, validation_steps=self.ds.test_len//32, callbacks=[self.logger, self.checkpoint])
        self.history = history