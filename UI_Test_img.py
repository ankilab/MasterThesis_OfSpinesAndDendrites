from PyQt5.QtWidgets import QWidget
import sys
import os
import numpy as np
from random import randrange
from skimage import io
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QDesktopWidget, QGridLayout, QRadioButton, QMessageBox
from PyQt5 import QtGui
import pandas as pd

base_path = 'D:/jo77pihe/Registered/20220223_Deconv_comp'
aq = os.path.join(base_path, '20220203_AutoQuant_NotAveraged')
care = os.path.join(base_path, 'CARE')
blind = os.path.join(base_path, 'BlindRL')
mu3 = os.path.join(base_path, 'MuNet')

names = ['AQ', 'CARE', 'Blind_RL', 'Mu_Net']

paths = [aq,care,blind,mu3]

num = 20
files = [f for f in os.listdir(aq) if f.endswith('.tif')]

MAX_VAL = 12870
MIN_VAL = -2327


class ImageSelection(QWidget):
    """
    User Interface for blind comparison of images. The same image plane is shown deconvolved with two different
    algorithms. The user is asked to select the plane he/she likes better. In the end, statistics are returned that
    reflect on how often each deconvolution method was selected over the other ones.

    """

    def __init__(self):
        """
        Initialize user interface.
        """

        super().__init__()

        self.res = np.zeros((len(names), len(names)))
        self.counter = np.zeros((len(names), len(names)))
        self.cx = False
        self.prev_a = 0
        self.prev_b = 0

        self.center()
        self.grid = QGridLayout()
        self.grid.setSpacing(5)
        self.imgA = QLabel(self)
        self.imgB = QLabel(self)
        self.random_images()
        self.imgA.show()
        self.imgB.show()

        self.grid.addWidget(self.imgA, 2, 1, 2, 2)
        self.grid.addWidget(self.imgB, 2, 3, 2, 2)

        self.lab1 = QLabel("Image A")
        myFont = QtGui.QFont()
        myFont.setBold(True)
        self.lab1.setFont(myFont)
        self.grid.addWidget(self.lab1,1,1,1,1)

        self.lab2 = QLabel("Image B")
        self.lab2.setFont(myFont)
        self.grid.addWidget(self.lab2,1,3,1,1)

        self.lab2 = QLabel("Select the better image.")
        self.grid.addWidget(self.lab2,5,1,1,1)

        self.btn = QPushButton('Next', self)
        self.grid.addWidget(self.btn, 6,2,5,1)
        self.btn.clicked.connect(self.show_img)
        self.rb1 = QRadioButton('Nothing selected')
        self.rb1.setChecked(True)
        self.grid.addWidget(self.rb1, 6,1,1,1)

        self.rb2 = QRadioButton("Image A")
        self.grid.addWidget(self.rb2,7,1,1,1)

        self.rb3 = QRadioButton("Image B")
        self.grid.addWidget(self.rb3,8,1,1,1)

        self.setGeometry(100, 100, 700, 500)
        self.setWindowTitle('Image comparison')
        self.setLayout(self.grid)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def show_img(self):
        """
        Display images.
        """

        # Count selection to stastics
        if self.cx:
            if self.rb2.isChecked():
                x = self.rb2.text()
            elif self.rb3.isChecked():
                x = self.rb3.text()
            else:
                x = None

            # x = await wait_for_change(chkbox)
            # print(x)
            self.counter[self.prev_a, self.prev_b] += 1
            self.counter[self.prev_b, self.prev_a] += 1
            if x == 'Image A':
                self.res[self.prev_a, self.prev_b] += 1
            elif x =='Image B':
                self.res[self.prev_b, self.prev_a] += 1

        self.random_images()
        self.rb1.setChecked(True)

    def random_images(self):
        """
        Select random image planes and deconvolution techniques.
        """

        # show new images
        a = randrange(len(names))
        b = randrange(len(names))
        c = randrange(len(files))

        if a != b:
            f1 = os.path.join(paths[a], files[c])
            f2 = os.path.join(paths[b], files[c])
            if os.path.isfile(f1) and os.path.isfile(f2):
                img1 = io.imread(os.path.join(paths[a], files[c]))
                img2 = io.imread(os.path.join(paths[b], files[c]))
                plane = randrange(img2.shape[0])
                img1 = img1[plane,:,:]
                img2=img2[plane,:,:]

                self.imgA.setPixmap(self.img_2_pixmap(img1))
                self.imgB.setPixmap(self.img_2_pixmap(img2))

                self.prev_a = a
                self.prev_b = b
                self.cx = True
            else:
                self.cx = False
                self.random_images()
        else:
            self.cx = False
            self.random_images()

    def img_2_pixmap(self, img):
        """
        Convert image to displayable version.

        :param img: Input image
        :type img: nd.array
        :return: Displayable image
        :rtype: nd.array
        """
        img = (self._rescale(img) * 255).astype(np.uint8)
        img = QtGui.QImage(img, img.shape[0], img.shape[0], img.shape[0], QtGui.QImage.Format_Indexed8)
        return QtGui.QPixmap.fromImage(img)

    def _rescale(self, img):
        """
        Rescale input to range of 0 and 1

        :param img: Input
        :type img: nd.array
        :return: Rescaled input
        :rtype: nd.array
        """

        img = (img - img.min())
        return img / img.max()

    def closeEvent(self, event):
        """
        Specify behaviour upon closing the application. The statistics are saved to file.

        :param event: Close event
        """
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            np.save('counter_comparison.npy', self.counter)
            np.save('res_comparison.npy', self.res)

            event.accept()
        else:
            event.ignore()


def get_result():
    """
    Load statistics.

    :return: Statistics
    :rtype: pd.DataFrame
    """
    counter=np.load('counter_comparison.npy')
    res = np.load('res_comparison.npy')
    x = (res/counter)
    df = pd.DataFrame(x, index=names, columns=names)
    return df


def main():
    """
    Run application.

    """
    app = QApplication(sys.argv)
    ex = ImageSelection()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
