from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from image_handler import ImageHandler


batch_size = 32
weights_path = 'output/weights_inception.hdf5'

labels, data = ImageHandler.prepare_images('images')

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=.25, random_state=42)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

image_data_generator = ImageDataGenerator(horizontal_flip=True)
train_generator = image_data_generator.flow(train_x, train_y, batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), padding='same'))
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

for i in range(1, 12):
    print(i)
    print(1 - model.predict(ImageHandler.prepare_image(ImageHandler.open_image(f'test/cats/{i}.jpg')))[0][0])

for i in range(0, 11):
    print(i)
    print(model.predict(ImageHandler.prepare_image(ImageHandler.open_image(f'test/others/{i}.jpg')))[0][0])

#fit_history = model.fit(train_generator, epochs=25, validation_data=(test_x, test_y))

#model.save_weights('output/weights_inception.hdf5')

#print(model.predict(ImageHandler.prepare_image(ImageHandler.open_image('031.jpg')), batch_size=batch_size))
