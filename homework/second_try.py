import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from image_handler import ImageHandler


batch_size = 32
weights_path = 'output/weights_inception.hdf5'

labels, data = ImageHandler.prepare_images('images')

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=.25)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

train_x = np.reshape(train_x, tuple([*train_x.shape] + [1]))
test_x = np.reshape(test_x, tuple([*test_x.shape] + [1]))
train_y = np.reshape(train_y, tuple([*train_y.shape] + [1]))
test_y = np.reshape(test_y, tuple([*test_y.shape] + [1]))

print(train_x.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

if weights_path is not None:
    model.load_weights(weights_path)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

#fit_history = model.fit(train_x, train_y, epochs=25, validation_data=(test_x, test_y))

#model.save_weights('output/weights_inception.hdf5')

for i, filename in enumerate(os.listdir('test/cats')):
    image = ImageHandler.prepare_image(ImageHandler.open_image(f'test/cats/{filename}'))
    prediction = 1 - model.predict(image)[0][0]
    ImageHandler.save_result(ImageHandler.open_image(f'test/cats/{filename}'),
                             f'{prediction * 100:.2f}%',
                             f'cat{i}.jpg')

for i, filename in enumerate(os.listdir('test/others')):
    image = ImageHandler.prepare_image(ImageHandler.open_image(f'test/others/{filename}'))
    prediction = 1 - model.predict(image)[0][0]
    ImageHandler.save_result(ImageHandler.open_image(f'test/others/{filename}'),
                             f'{prediction * 100:.2f}%',
                             f'other{i}.jpg')
