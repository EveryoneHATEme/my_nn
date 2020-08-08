from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Dense, Flatten, Activation
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np

import pickle

from image_handler import ImageHandler

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class VGG16:
    def __init__(self, input_shape, number_of_classes, model_path=None):

        if model_path:
            self.model = load_model(model_path)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(.25))

            self.model.add(Conv2D(64, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(64, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(.25))

            self.model.add(Conv2D(128, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(128, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(128, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(.25))

            self.model.add(Flatten())
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(.25))

            self.model.add(Dense(number_of_classes))
            self.model.add(Activation('sigmoid'))

            self.optimizer = None
            self.label_binarizer = LabelBinarizer()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def compile(self, loss='binary_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']

        self.model.compile(self.optimizer, loss, metrics)

    def train(self, batch_size=32, epochs=25, plot=True):
        print('loading images')
        labels, data = ImageHandler.prepare_images('images')

        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

        train_y = self.label_binarizer.fit_transform(train_y)
        test_y = self.label_binarizer.transform(test_y)

        image_generator = ImageDataGenerator(rotation_range=30, width_shift_range=.1, height_shift_range=.1,
                                             shear_range=.2, zoom_range=.2, horizontal_flip=True, fill_mode='nearest')

        print('start training')
        fit_history = self.model.fit_generator(image_generator.flow(train_x, train_y, batch_size),
                                               len(train_x) // batch_size, epochs=epochs,
                                               validation_data=(test_x, test_y))

        print('finish training')
        if plot:
            self.plot(epochs, fit_history)

    def predict(self, image_path):
        original_image = ImageHandler.open_image(image_path)
        image = ImageHandler.prepare_image(original_image)
        prediction = self.model.predict(image)[0][0]
        text = f'cat: {prediction * 100:.2f}%' if prediction > 0.5 else f'not a cat: {prediction * 100:2f}%'
        ImageHandler.show_result(original_image, text)

    @staticmethod
    def plot(epochs, fit_history):
        epochs_array = np.arange(epochs)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(epochs_array, fit_history.history['loss'], label='train loss')
        plt.plot(epochs_array, fit_history.history['accuracy'], label='train_accuracy')
        plt.title('Training loss and accuracy')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.savefig('output/plot.png')

    def save(self, direction):
        self.model.save(direction)
        with open('output/label_binarizer', 'wb') as f:
            f.write(pickle.dumps(self.label_binarizer))


NN = VGG16((64, 64, 3), 1, 'output/nn.model')
NN.predict('w2Cg3wehJQI.jpg')

# learning_rate = 0.01
# epochs = 25
# batch_size = 32
# NN.set_optimizer(SGD(learning_rate=learning_rate, decay=learning_rate / epochs))
# NN.compile()
# NN.train()
# NN.save('output/nn.model')
