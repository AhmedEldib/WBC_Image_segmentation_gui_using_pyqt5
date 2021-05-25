from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem, QMessageBox , QSlider , QLabel
from pyqtgraph import PlotWidget, image, plot, PlotItem
import pyqtgraph as pg
from gui import Ui_MainWindow

from Cmeans import CMeans 
from shallow_network_segmentation import SNN

from SVM import svm_class

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io 
from skimage.transform import resize
from skimage import exposure
from skimage.color import rgb2gray

import logging
import sys
import os
import warnings

#python -m PyQt5.uic.pyuic -x gui.ui -o gui.py

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        pg.setConfigOption('background', 'w')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.imported_image = None
        self.snnImage = None
        self.svmImage = None

        self.no_clusters = None
        self.cluster = None

        self.ui.import_button.clicked.connect(lambda: self.Import(0))
        self.ui.pushButton.clicked.connect(self.Slider)
        self.ui.pushButton_3.clicked.connect(self.shallow_network)
        self.ui.pushButton_2.clicked.connect(lambda: self.Import(1))
        self.ui.pushButton_5.clicked.connect(lambda: self.Import(2))
        self.ui.pushButton_4.clicked.connect(self.svm)

#////////////////////////////// GUI Functions /////////////////////////////////////
#------------------------------------------------------------------------------------------------------------------------
    def plot(self, Graph, img):
        Graph.clear()
        Graph.addItem(pg.ImageItem(img))

        self.View_size(0, img.shape[0], 0, img.shape[1], Graph)
        QtCore.QCoreApplication.processEvents()

    def View_size(self, Xmini, Xmaxi, Ymini, Ymaxi, imgGraph):
        imgGraph.plotItem.getViewBox().setLimits(xMin=Xmini - 50, xMax=Xmaxi + 50, yMin=Ymini - 50, yMax=Ymaxi + 50)

    def Slider(self):
        sliderValue = self.ui.classes_slider.value()

        if self.imported_image is not None:
            self.get_clusters(sliderValue)            

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("No Imported image found")
            msg.exec_()

        return


    def Import(self, num):
        filePaths = QtWidgets.QFileDialog.getOpenFileNames(self, 'Multiple File',"~/Desktop",'*.jpg && *.png && *.bmp')
        for filePath in filePaths:
            for f in filePath:
                print('filePath',f, '\n')
                if f == '*' or f == None:
                    break
                
                temp = io.imread(f)
                if (len(temp) != 0):
                    if temp.shape[2] == 4:
                        temp = np.delete(temp, 3, 2)

                    self.imported_image = temp #resize(temp, (88, 88), anti_aliasing=True, preserve_range=True)

                    flipped_img = np.rot90(self.imported_image, 3)

                    if num == 0:
                        self.cMeansImage = np.copy(self.imported_image)
                        self.plot(self.ui.original_image, flipped_img)
                    elif num ==1:
                        self.snnImage = np.copy(self.imported_image)
                        self.plot(self.ui.graphicsView, flipped_img)
                    elif num ==2:
                        self.svmImage = np.copy(self.imported_image)
                        self.plot(self.ui.graphicsView_3, flipped_img)
#------------------------------------------------------------------------------------------------------------------------


#////////////////////////////// Segmentation Models Functions /////////////////////////////////////
#------------------------------------------------------------------------------------------------------------------------
    def shallow_network (self):
        if self.snnImage is not None:
            imagesegmented, report =SNN(self.snnImage)
            imagesegmented = np.rot90(imagesegmented, 3)
            self.plot(self.ui.graphicsView_2, imagesegmented)
            self.ui.textBrowser.setText("Test accuracy of the Neural Network during training was: \n \n " +str(report))            

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("No Imported image found")
            msg.exec_()

        return       

                    
    def svm (self):
        if self.svmImage is not None:
            imagesegmented, report =svm_class(self.svmImage)
            imagesegmented = np.rot90(imagesegmented, 3)
            self.plot(self.ui.graphicsView_4, imagesegmented)
            self.ui.textBrowser_2.setText("Test accuracy of the SVM during training was: \n \n " + str(report))            

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("No Imported image found")
            msg.exec_()

        return


    def get_clusters(self,C=2):
        pixel_values = self.cMeansImage.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        model = CMeans(C, max_iters=10)  
        y_pred = model.predict(pixel_values) 
        centers = np.uint8(model.cent())

        y_pred = y_pred.astype(int)
        labels = y_pred.flatten()

        imagesegmented = centers[labels.flatten()]
        imagesegmented = imagesegmented.reshape(self.cMeansImage.shape)

        flipped_img = np.rot90(imagesegmented, 3)

        self.plot(self.ui.segmented_image, flipped_img)


    #------------------------------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------------------------

#////////////////////////////// Main /////////////////////////////////////

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()