from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.preprocessing.image import img_to_array, load_img

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.losses import categorical_crossentropy
import cv2

#--------------------------------------------Importar imágenes-------------------------------------------

#código para cargar muchas imágenes de un solo directorio. Tiene que estar en un subcarpeta local, y cada
#carpeta de dentro tiene que tener imágenes. Cada imagen será entonces de la clase de una de las carpetas

img_width = 128
img_height = 64

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'Database ALERT v2\Database ALERT\cam32',  # this is the target directory
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
#SE PONE CATEGORICAL PORQUE LA FUNCION ES CATEGORICAL_CROSSENTROPY: One-hot para varias clases, wey

# this is the augmentation configuration we will use for testing:
# only rescaling

#nota: test_datagen va con test_generator. En vez de poner "test" poner "validation" pej da error
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
        'Database ALERT v2\Database ALERT\cam37',
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode='categorical')

img = train_generator[1]



#---------------------------------Costrucción del modelo-----------------------------------------

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(3, img_width, img_height)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1505, activation='softmax'))
#lo de 1506 es el nº de clases de la cam32 -> cambiar si eso
#Y ES NECESARIO SABER EL Nº DE CLASES DEL TRAINING SET. Si solo hubiese dos, con el binary valdría.
#Prueba 3: probar con el mayor nº de clases posibles.

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#---------------------------------------Entrenamiento y fitteado------------------------------------

print('model complied!!')

#activar esto cuando ya se ha entrenado

#model.load_weights("red2_pesos_doscams.h5")
#model.load_weights("red2_pesos_batchgrande.h5")
#model.load_weights("red2_pesos_batchtestreducido.h5")


print('starting training....')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=test_generator,
        validation_steps=800 // batch_size,
        verbose=0)
model.save_weights('red2_pesos_doscams.h5')  # always save your weights after training or during training

print('training finished!!')


print('all weights saved successfully !!')
#models.load_weights('models/simple_CNN.h5')

print('trying to evaluate thingies')

scores = model.evaluate_generator(test_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
print('%s : %s' % (model.metrics_names[0], scores[0]))
print('%s : %s' % (model.metrics_names[1], scores[1]))
