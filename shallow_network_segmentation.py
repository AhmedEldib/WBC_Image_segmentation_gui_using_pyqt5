import numpy as np 
import pandas as pd
import pickle 

from skimage import io 
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import os
from network import ShallowNeuralNetwork

def SNN (image):

    x = []
    y = []

    path = '/train'
    filepaths = [os.path.join(r,file) for r,d,f in os.walk(os.getcwd() + path) for file in f]
    filepaths = [x for x in filepaths if x.endswith(".PNG")]

    for filename in filepaths:
        im = io.imread(filename)
        im = np.round((resize(im, (36, 36), anti_aliasing=True, preserve_range=True))).astype(int)

        if im.shape[2] == 4:
            im = np.delete(im, 3, 2)

        if filename.split("_")[-1].split(".")[0] == '1':
            x.append(im)
            y.append(1)

        elif filename.split("_")[-1].split(".")[0] == '2':
            x.append(im)
            y.append(2)

        elif filename.split("_")[-1].split(".")[0] == '3':
            x.append(im)
            y.append(3)

    X_train = np.array(x)
    labels = np.array(y)
    len(X_train)

    #X_train = np.delete(X_train, 3, 3)

    x = np.copy(X_train)
    # x[0].shape

    pixel_values = []

    for i in x:
        pixel_values.append(np.float32(i)/255)

    x = np.array(pixel_values)

    np.insert(x[0].reshape(-1, 3), 3, 1, axis=1)

    #x[0].reshape(-1, 3).shape

    data = pd.DataFrame(columns = ["r", "g", "b", "class"])

    for i in range(len(labels)):
        data = data.append(pd.DataFrame(np.insert(x[i].reshape(-1, 3), 3, labels[i], axis=1), columns = ["r", "g", "b", "class"]))


    data = data.reset_index()
    data = data.drop(["index"], axis=1)

    x = data.iloc[:, [0, 1, 2]].values.astype(np.float64)

    y = data['class'].values

    lb = LabelBinarizer()
    labels = data['class'].values
    labels = lb.fit_transform(labels)

    X_tr, X_te, Y_tr, Y_te = train_test_split(x, labels, test_size=0.33, shuffle=True, random_state=49)

    # snn = ShallowNeuralNetwork(sizes=[3, 8, 3], l_rate=0.01)
    # snn.train(X_tr, Y_tr, X_te, Y_te)

    with open('snn.pickle', 'rb') as f:     
        snn = pickle.load(f)

    y_p = []

    for i in X_te:
        k = snn.predict(i)
        y_p.append(np.argmax(k))

    y_actual = []

    for i in Y_te:
        y_actual.append(np.argmax(i))

    X_test = image
    l, w = X_test.shape[0:2]

    pixel_values = []

    pixel_values = np.float32(X_test)/255

    X_test = np.array(pixel_values)


    test = pd.DataFrame(columns = ["r", "g", "b"])
    test = test.append(pd.DataFrame(X_test.reshape(-1, 3), columns = ["r", "g", "b"]))

    y_pred = []

    for i in test.values:
        y_pred.append(snn.predict(i).argmax())

    y_pred = np.array(y_pred)

    np.unique(y_pred, return_counts=True)

    cluster1_mean = np.mean(test[y_pred == 0], axis=0)
    cluster2_mean = np.mean(test[y_pred == 1], axis=0)
    cluster3_mean = np.mean(test[y_pred == 2], axis=0)

    y_pred.reshape(-1, 1)

    y_mean = []
    for i in y_pred:
        if i == 0:
            y_mean.append(cluster1_mean)

        elif i == 1:
            y_mean.append(cluster2_mean)

        elif i == 2:
            y_mean.append(cluster3_mean)
    y_mean = np.array(y_mean)


    np.unique(y_pred, return_counts=True)

    y_mean = y_mean.reshape(l, w, 3)

    return y_mean , (classification_report(y_p, y_actual))

    



