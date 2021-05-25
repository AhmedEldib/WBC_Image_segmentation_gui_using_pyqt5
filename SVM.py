import numpy as np 
import pandas as pd 
import pickle

from skimage import io 
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import os

def svm_class(image):

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

        if filename.split("_")[1].split(".")[0] == '1':
            x.append(im)
            y.append(1)

        elif filename.split("_")[1].split(".")[0] == '2':
            x.append(im)
            y.append(2)

        elif filename.split("_")[1].split(".")[0] == '3':
            x.append(im)
            y.append(3)


    X_train = np.array(x)
    labels = np.array(y)
    len(X_train)

    #X_train = np.delete(X_train, 3, 3)

    x = np.copy(X_train)

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

    x = data.iloc[:, [0, 1, 2]].values

    y = data['class'].values

    X_tr, X_te, Y_tr, Y_te = train_test_split(x, y, test_size=0.33, shuffle=True, random_state=49)

    # # parameter_candidates = [
    # #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    # #   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    # # ]

    # # # Create a classifier with the parameter candidates
    # # clf = GridSearchCV(estimator = svm.SVC(), param_grid = parameter_candidates, n_jobs = -1)

    # # # Print out the results 
    # # print('Best score for training data:', clf.best_score_)
    # # print('Best `C`:',clf.best_estimator_.C)
    # # print('Best kernel:',clf.best_estimator_.kernel)
    # # print('Best `gamma`:',clf.best_estimator_.gamma)

    # clf = svm.SVC(C=100, kernel='linear', gamma='scale')

    # # Train the classifier on training data
    # clf.fit(X_tr, Y_tr) 

    with open('clf.pickle', 'rb') as f:     
        clf = pickle.load(f)

    X_test = image
    l, w = X_test.shape[0:2]

    pixel_values = []

    pixel_values = np.float32(X_test)/255

    X_test = np.array(pixel_values)



    test = pd.DataFrame(columns = ["r", "g", "b"])
    test = test.append(pd.DataFrame(X_test.reshape(-1, 3), columns = ["r", "g", "b"]))

    X_test = test.iloc[:, :].values

    y_pred = clf.predict(X_test)

    cluster1_mean = np.mean(X_test[y_pred == 1], axis=0)
    cluster2_mean = np.mean(X_test[y_pred == 2], axis=0)
    cluster3_mean = np.mean(X_test[y_pred == 3], axis=0)

    y_pred.reshape(-1, 1)

    y_mean = []
    for i in y_pred:
        if i == 1:
            y_mean.append(cluster1_mean)

        elif i == 2:
            y_mean.append(cluster2_mean)

        elif i == 3:
            y_mean.append(cluster3_mean)
    y_mean = np.array(y_mean)


    np.unique(y_pred, return_counts=True)

    y_mean = y_mean.reshape(l, w, 3)

    #accuracy
    y_p = clf.predict(X_te)

    return y_mean , (classification_report(y_p, Y_te))

    


