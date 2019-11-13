import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, Activation, LeakyReLU, Softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

def generate_model():
    vgg = load_model("vgg16.h5")

    new_model = Sequential()
    for layer in vgg.layers[:20]:
        layer.trainable = False
        new_model.add(layer)

    new_model.add(Dense(64))
    new_model.add(LeakyReLU(0.1))
    new_model.add(Dense(20))
    new_model.add(Softmax())
    
    return new_model

def get_generators():
    datagen = ImageDataGenerator(
        rescale = 1/255.,
        )

    train_generator = datagen.flow_from_directory(
        "bird_dataset/bird_dataset/train_images",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical")

    val_generator = datagen.flow_from_directory(
        "bird_dataset/bird_dataset/val_images",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical")

    return train_generator, val_generator

def main():
    model = generate_model()
    traingen, valgen = get_generators()

    model.compile(Adam(lr=1e-4), categorical_crossentropy,
                  metrics=["accuracy"])

    model.fit_generator(generator=traingen,
                        steps_per_epoch=34,
                        epochs=10,
                        validation_data=valgen,
                        validation_steps=4)

if __name__=="__main__":
    main()
