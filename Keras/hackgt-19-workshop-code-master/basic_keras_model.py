# in python
from keras.models import *

# creates a "sequential" type model, with each layer coming after the last one
model = Sequential()

from keras.layers import *

# we add each layer one by one
model.add(Dense(3, input_shape=(2,), use_bias=True)) # need input_shape here
model.add(Activation("sigmoid"))
model.add(Dense(1, use_bias=True))  # only need input_shape on first layer
model.add(Activation("sigmoid"))
# activation functions are important! in lecture I forgot them at first.

model.summary()
# just prints out a nice text block that displays your model

from keras.optimizers import *

model.compile(SGD(lr=0.001), loss="mean_squared_error")
# tells keras what to use as the optimizer (gradient desecent) and
# what objective to minimize (known as loss function, in this case, it's
# mean squared error)


# now we need some data to train on

import numpy as np

# create the X matrix (inputs)
X = np.zeros((4, 2)) # 4 training examples, each has 2 features (x0, x1)
X[0] = (0, 0)
X[1] = (1, 0)
X[2] = (0, 1)
X[3] = (1, 1)

# create the y matrix (labels)
y = np.zeros((4, 1)) # 4 training examples, each has 1 label
y[0] = 0
y[1] = 1
y[2] = 1
y[3] = 0

# now we can train the model
model.fit(X, y, epochs=100)

# For some reason, the loss won't go to 0, probably because I didn't tune
# one of the parameters correctly. I'll have a working example for you guys
# next week, but in general it's going to look something like this code.








