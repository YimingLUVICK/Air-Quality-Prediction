from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def model():
    m = Sequential()

    m.add(LSTM(64, input_shape=(50, 17), return_sequences=True))
    m.add(LSTM(32))
    m.add(Dense(1, activation='sigmoid')) 

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return m

    