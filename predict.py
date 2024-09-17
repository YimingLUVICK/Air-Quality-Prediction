import numpy as np
import tensorflow as tf

def predict(model, ds):
    y_pred = []
    y_true = []
    for i in range(ds.test_len):
        x = np.array(ds.df.iloc[ds.X_test[i]:ds.X_test[i]+50])
        x = np.expand_dims(x, 0)
        y = ds.y_test[i]
        p = model.predict(x)
        if p>=0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(int(y))
    return y_pred, y_true