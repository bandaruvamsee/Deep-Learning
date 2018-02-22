# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:37:51 2018

@author: Sreenivas.J
"""

#History
#History gives you acc(Train accuracy), loss(Train loss), val_acc(validation data accuracy), val_loss(Validation data loss)
#This diff. b/w train_acc and val(idation)_acc gives you the feel of overfitting/underfitting. The diff. b/w train_acc and val(idation)_acc should be minimal.

#In DL, we anlayze the the data epoch by epoch. It's not step by step. Epoch by epoch takes some time b/w each epoch and anlyze further
#After each epoch, call back mechanism function keeps the best so far and ignore the non best calls. This way only the best epoch model will be retained.

#Verbose = 1 or 2 or 3
#Eg: 3 Gives detailed information epoch by epoch, Try with 1 and 2 options.

#Keep Keras documentation and this code side by side and you will the get the CRUX of the Keras and Sequential Model.

from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()

def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))


X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, \
                           n_informative=2, random_state=0, n_clusters_per_class=1)
print(X.shape)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=100)
print(X_train.shape)
print(X_validation.shape)

plot_data(X_train, y_train)

#perceptron model for binary classification
model = Sequential()
#input_shape means no.of features
#Units means no.of Dense units.
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, verbose=3, epochs=50, validation_data=(X_validation,y_validation), batch_size=10)
print(model.summary())
print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)

plot_loss_accuracy(history)

y_pred = model.predict_classes(X, verbose=0)
