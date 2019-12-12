from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import os


def get_test(batch_size):
    test = np.zeros([12500, 128, 128, 3])
    for i in range(12500//batch_size):
        test[i*batch_size:(i+1)*batch_size, :, :, :] = test_generator.next()
    return test




train_file_path = 'C:/Users/Lenovo/Desktop/dog_cat/dataset_kaggledogvscat/train'
test_file_path = 'C:/Users/Lenovo/Desktop/dog_cat/dataset_kaggledogvscat/test'
validation_file_path = 'C:/Users/Lenovo/Desktop/dog_cat/dataset_kaggledogvscat/validation'
output_path = 'C:/Users/Lenovo/Desktop/dog_cat/dataset_kaggledogvscat/class.csv'


idg = ImageDataGenerator()
train_generator = idg.flow_from_directory(train_file_path, (128, 128), shuffle=True, batch_size=100, class_mode='categorical')
validation_generator = idg.flow_from_directory(validation_file_path, (128, 128), shuffle=True, batch_size=100, class_mode='categorical')
test_generator = idg.flow_from_directory(test_file_path, (128, 128), shuffle=False, batch_size=100, class_mode=None)

test = get_test(100)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(128, 128, 3), activation='relu', kernel_initializer='uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, epochs=35, steps_per_epoch=23000//100, validation_data=validation_generator, validation_steps=10)


pre = model.predict_classes(test, batch_size=100, verbose=1)
cla = []
for i in range(len(pre)):
    if pre[i] == 0:
        cla.append('cat')
    else:
        cla.append('dog')
df = DataFrame({'class': cla})
df.to_csv(output_path)
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Training acc')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()


