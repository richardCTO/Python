#%%
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# One hot encodes, turns values like 5
# into a vector of 0s and 1s
# 5 -> [0,0,0,0,0,1,0,0,0,0]
def one_hot_encode(values, num_classes):
    return np.eye(num_classes)[values]

# Turns y vectors into one hot encoded vectors
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)

# Flattens our X data so a fully connected
# neural network can take it as input
x_train = np.reshape(x_train, (60000, 28 * 28))
x_test = np.reshape(x_test, (10000, 28 * 28))

# Sequential model
model = Sequential()
model.add(Dense(32, input_shape=(28*28,))) 
# Input shape is the same thing that we just
# resized it to.

model.add(LeakyReLU(0.1))
# Activation function, nonlinearity

model.add(Dense(10))
# Output layer

model.add(Softmax())
# Softmax activation function required at the end
# for multiclass classification (num_classes > 2)

# print summary of model for our benefit
model.summary()

model.compile(optimizer=Adam(lr=0.001), \
              loss=categorical_crossentropy, \
              metrics=["accuracy"]
              )
# compile with Adam optimizer and categorical
# cross entropy loss (which is what we use for
# multiclass classification)

model.fit(x=x_train, y=y_train, batch_size=32, \
          epochs=10, validation_data=(x_test, y_test))


# %%
