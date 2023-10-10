# -*- coding:utf-8 -*-
import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import matthews_corrcoef,confusion_matrix, accuracy_score
from SRPNet.process import StrokeDataset,featureselect,set_seed
from SRPNet.model import TransformerModel,SRPNetModel,CNN,LSTM
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import argparse


def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='inhouse')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=20000, type=int)
    args = parser.parse_args()
    set_seed()
    return args

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts = parser_set()

    if opts.dataset=="CSDC":
        data_name = '26feature'
        file_name = 'data/' + data_name + '.csv'
        sname = 'data/sample-' + data_name + '.csv'
        data_x = pd.read_csv(file_name,header=0,index_col=0)
        print(data_x['risk'].value_counts())
        y = data_x['risk'].values - 1
        X = data_x.drop('risk', axis=1)
        num_classes=5
        risk = ["Low risk", "Medium risk", "High risk", "TIA", "Stroke"]


    elif opts.dataset=="CSDC_selected":
        data_name = '2011stroke'
        file_name = 'data/' + data_name + '.csv'
        sname = 'data/sample-' + data_name + '.csv'
        data_x = pd.read_csv(file_name)
        y = data_x['risk'].values - 1
        data_x=featureselect(data_x)
        X = data_x[["Smoking"," Occupation", "ES", "HS", "HYP", "AF","LDL-C",
                  "Diabetes", "LE", "Overweight", "FHS/HYP/CHD","PLT"]]
        num_classes=5
        risk = ["Low risk", "Medium risk", "High risk", "TIA", "Stroke"]

    elif opts.dataset=="inhouse":
        data_x = pd.read_csv("data/stroke2023.csv")
        y = data_x['risk'].values
        X = data_x.drop('risk', axis=1)
        num_classes=2
        risk = ["health", "stroke"]

    print(X.columns.values)
    print(X.shape)
    # X.to_csv("f12.csv")
    X=X.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    leny=len(y_test)
    print(leny)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_dataset = StrokeDataset(X_train,  y_train,device)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=0)
    test_dataset = StrokeDataset(X_test,  y_test,device)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=0)

    input_size = X_train.shape[1]

    model = SRPNetModel(input_size=input_size, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr)

    print("train model")


    for epoch in range(opts.num_epochs):
        for i, tensor_data in enumerate(train_loader):
            inputs, labels = tensor_data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{opts.num_epochs}], Loss: {loss.item():.4f}')

    print("test model")
    pred_batches=[]
    label_batches=[]
    with torch.no_grad():

        for i, tensor_data in enumerate(test_loader):
        # 对测试数据集进行预测，并与真实标签进行比较，获得预测
            inputs, labels = tensor_data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            pred_batches.append(predicted.cpu().numpy())
            label_batches.append(labels.cpu().numpy())
            # print(acc)
        # 拼接预测结果
        y_pred = np.concatenate(pred_batches, axis=0)
        y_true = np.concatenate(label_batches, axis=0)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm / np.sum(cm,axis=0)
    print(cm)
    # 计算MCC

    print('------Weighted------')
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
    print('------Macro------')
    print('Macro precision', precision_score(y_true, y_pred, average='macro'))
    print('Macro recall', recall_score(y_true, y_pred, average='macro'))
    print('Macro f1-score', f1_score(y_true, y_pred, average='macro'))
    print('------Micro------')
    print('Micro precision', precision_score(y_true, y_pred, average='micro'))
    print('Micro recall', recall_score(y_true, y_pred, average='micro'))
    print('Micro f1-score', f1_score(y_true, y_pred, average='micro'))
    ka=cohen_kappa_score(y_true, y_pred)
    print(opts.dataset,"cohen_kappa_score: ",ka)

    fig, ax = plt.subplots()
    im = ax.imshow(cm,cmap="Oranges")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(risk)), labels=risk, fontsize=14)
    ax.set_yticks(np.arange(len(risk)), labels=risk, fontsize=14)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(risk)):
        for j in range(len(risk)):
            if cm[i, j]<=0.6:
                text = ax.text(j, i,"%.2f%%" % (cm[i, j]*100),
                               ha="center", va="center", color="black", fontsize=14)
            else:
                text = ax.text(j, i, "%.2f%%" % (cm[i, j]*100),
                               ha="center", va="center", color="w", fontsize=14)
    ax.set_title("{} Confusion Matrix(SRPNet)".format(opts.dataset), fontsize=14)
    fig.tight_layout()
    plt.savefig('result/SRPNet_{}.pdf'.format(opts.dataset), bbox_inches='tight')
    plt.show()

