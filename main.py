import platform
from flask import Flask, render_template, redirect, request, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from os.path import splitext
import threading
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from kt_utils import *
import tensorflow as tf
import keras.losses as losses
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing import image

if int(platform.python_version_tuple()[1]) >= 6:
    import asyncio
    import asyncio.base_futures
    import asyncio.base_tasks
    import asyncio.compat
    import asyncio.base_subprocess
    import asyncio.proactor_events
    import asyncio.constants
    import asyncio.selector_events
    import asyncio.windows_utils
    import asyncio.windows_events

    import jinja2.asyncsupport
    import jinja2.ext

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/images'
configure_uploads(app, photos)

model_1 = load_model('classifier.h5')
model_2 = load_model('final_model.h5')
model_3 = load_model('classifier_noise.h5')


def predict_words(filename, model, title):
    img = cv2.imread("static\\images\\{y}".format(y=filename))
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for y in range(0, height, 76):
        for x in range(0, width, 111):
            crop_img = cv2.resize(img[y:y + height, x:x + width], (110, 120))
            test_image = image.img_to_array(crop_img)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255
            result = model.predict(test_image)
            result = result[0]
            print(result)
            for i in range(0, len(result)):
                if result[i] >= 0.95:
                    result[i] = 1
                else:
                    result[i] = 0
            if len(result) != 1:
                if result[0] == 0 and result[2] == 0:
                    result[1] = 1
                    print(result)
                if result[0] == 1:
                    rect = patches.Rectangle((x, y), 111, 76, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                elif result[2] == 1:
                    rect = patches.Rectangle((x, y), 111, 76, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
            else:
                if result == 0:
                    rect = patches.Rectangle((x, y), 111, 76, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle((x, y), 111, 76, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
    plt.title(title)
    plt.show()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ocr", methods=['POST'])
def ocr():
    file = request.files['input_file']
    filename = photos.save(file)
    print(filename)
    filename = 'sample.jpg'
    predict_words(filename, model_1, 'Just Main And Aur; Model 1')
    predict_words(filename, model_2, 'With noise; Model 2')
    predict_words(filename, model_3, 'With noise; Model 3')
    return render_template("index.html")


@app.route("/sample")
def sample():
    predict_words('sample.jpg', model_1, 'Model 1; Sample.')
    return render_template("index.html")


if __name__ == "__main__":
    thread = threading.Thread(target=app.run)
    thread.daemon = True
    thread.start()
    # app.run(debug=True)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("UrduOCR")
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("http://127.0.0.1:5000/"))
        self.setCentralWidget(self.browser)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
