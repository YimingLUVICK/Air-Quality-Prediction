import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def normalize(data, threshold):
    span = max(data) - min(data)
    return (data - threshold) / span

def get_df_datas(path, name, threshold):
    df_datas = pd.read_excel(path, sheet_name=name)
    
    df_datas['year'] = df_datas['Date'].dt.year
    df_datas['month'] = df_datas['Date'].dt.month
    df_datas['day'] = df_datas['Date'].dt.day
    df_datas['time'] = df_datas['Time'].apply(lambda x: x.hour)
    df_datas = df_datas.drop(['Date', 'Time'], axis=1)

    df_datas.replace(-200, np.nan, inplace=True)
    df_datas = df_datas.interpolate(method='linear')

    for e in df_datas:
        if e == 'CO(GT)':
            df_datas[e] = normalize(df_datas[e], threshold)
        else:
            mean = sum(df_datas[e]) / len(df_datas[e])
            df_datas[e] = normalize(df_datas[e], mean)

    return df_datas

def get_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

def get_acc_loss(log_path):
    df_log = pd.read_csv(log_path)
    x = df_log['Epoch']
    t_loss = df_log['Train Loss']
    t_acc = df_log['Train Accuracy']
    v_loss = df_log['Val Loss']
    v_acc = df_log['Val Accuracy']
    plt.figure(figsize=(8, 6))
    plt.plot(x, t_loss, label='train loss', color='blue')
    plt.plot(x, t_acc, label='train acc', color='green')
    plt.plot(x, v_loss, label='val loss', color='red')
    plt.plot(x, v_acc, label='val acc', color='purple')
    plt.legend()

    plt.savefig("process.png")
