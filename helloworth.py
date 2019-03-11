import tensorflow as tf

mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()  # introducimos el modelo
model.add(tf.keras.layers.Flatten())  # agregamos nuestro layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # agregamos las neuronas
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # agregamos las neuronas
model.add(tf.keras.layers.Dense(10, activation=tf.nn.log_softmax))  # output layer numero de clasificaciones

model.compile(optimizer='adam',  # agregamos nuestro optimizador
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('epic_num_reader.model')
new_model = tf.keras.model.load_model('epic_num_reader.model')
prediction = new_model.preict([x_test])
print(prediction)

import numpy as np

print(np.argmax(prediction[0]))

plt.imshow(x_test[0])
plt.show()


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
print(x_train[0])
