import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# 1st model is Sequential API (one input to one output)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())
# sys.exit()

# or we can define Sequential API in other format
model = keras.Sequential()
model.add(keras.Input(shape=(28 * 28)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(10))

# Functional API (multiple inputs to outputs, more flexible)
inputs = keras.Input(shape=(28 * 28))
x = layers.Dense(512, activation="relu", name="1st_layer")(inputs)
x = layers.Dense(256, activation="relu", name="2nd_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=32, verbose=2)
