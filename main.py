import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
# flattens the model from 28x28 to 1x784 array
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# layer with 10 nodes (10 possible outputs)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(x = x_train, y = y_train, epochs=5)

test_loss, test_acc = model.evaluate(x = x_test, y = y_test)
print("\nAccuracy: ", test_acc)

predictions = model.predict([x_test])
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))

plt.imshow(x_test[0], cmap="gray")
plt.show()
plt.imshow(x_test[1], cmap="gray")
plt.show()




