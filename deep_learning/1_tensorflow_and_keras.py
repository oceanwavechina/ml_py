import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

# 28x28 sized images of hand written digits 0-9
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train[0])
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# 现在数值 都是0-1之间的了
#print(x_train[0])

# build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

# dense layyer has 128 units
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# parameters for trainning
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# do the tranning
'''
model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)
print('loss && accuracy:', val_loss, val_acc)
model.save('epic_num_reader.model')
'''

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()

# display
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()