#!/usr/bin/env python3
"""
contains function preprocess_data
"""
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    preprocesses data for transfer learning model
    """
    x_prepro = K.applications.vgg16.preprocess_input(X)
    y_prepro = K.utils.to_categorical(Y)
    return x_prepro, y_prepro


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    xtrain_pre, ytrain_pre = preprocess_data(x_train, y_train)
    xtest_pre, ytest_pre = preprocess_data(x_test, y_test)
    inputs = K.Input(shape=(32, 32, 3))

    vgg = K.applications.VGG16(include_top=False, pooling='max',
                               input_tensor=inputs, weights='imagenet')
    output = vgg.get_layer('block3_pool').output
    j = K.layers.GlobalAveragePooling2D()(output)
    j = K.layers.BatchNormalization()(j)
    j = K.layers.Dense(256, activation='relu')(j)
    j = K.layers.Dropout(0.4)(j)
    j = K.layers.Dense(128, activation='relu')(j)
    j = K.layers.Dropout(0.2)(j)
    output = K.layers.Dense(10, activation='softmax')(j)
    model = K.Model(inputs=inputs, outputs=output)
    model.summary()

    reduc_learn_rate = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                     factor=.01, patience=3,
                                                     min_lr=1e-5)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(xtrain_pre, ytrain_pre,
                        validation_data=(xtest_pre, ytest_pre),
                        batch_size=128, callbacks=[reduc_learn_rate],
                        epochs=30, verbose=1)
    model.save('cifar10.h5')
