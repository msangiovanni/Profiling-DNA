# This file contains functions used for training the neural nets
# They call on functions in models.py

from math import exp
import numpy as np
from models import unet_small, FFN_DTDP
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * exp(-0.1)


def unet_train_test_split(train_input, test_input, length: int, weightpath='data/weights_NFI/weights_norm_avgpool.h5',
                          train=False, batchsize=10, epochs=10):
    """unet training. Default is training and storing under weights.h5
    @type train_input: object
    """
    number_of_dyes = 6
    model = unet_small((length, number_of_dyes, 1))
    model.summary()
    model.load_weights(weightpath)
    test_images = test_input.data
    test_labels = test_input.labels

    if train:
        import numpy
        train_images = numpy.asarray(train_input.data)
        train_labels = numpy.asarray(train_input.labels)
        batch_size = batchsize  # number of samples processed before the model is updated
        num_epochs = epochs  # number of complete passes through the training dataset before the training stops
        model_checkpoint = ModelCheckpoint(weightpath, monitor='val_loss', save_best_only=True)
        # history is optional for plotting
        import numpy as np
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
    metric_values = model.evaluate(x=test_images, y=test_labels)
    print('Final test performance')
    for metric_value, metric_name in zip(metric_values, model.metrics_names):
        print('{}: {}'.format(metric_name, metric_value))
    return model


def FFN(train_input, test_input, weightpath="data/weights_DTDP/weights_our_data_new.h5", inputsize=(1206,), train=False,
        batchsize=10, epochs=10):
    model = FFN_DTDP(input_size=inputsize)
    model.summary()
    # model.load_weights(weightpath)
    test_images = test_input.data
    test_labels = test_input.labels

    if train:
        train_images = train_input.data
        train_labels = train_input.labels
        batch_size = batchsize  # number of samples processed before the model is updated
        num_epochs = epochs  # number of complete passes through the training dataset before the training stops
        model_checkpoint = ModelCheckpoint(weightpath, monitor='val_loss', save_best_only=True)
        lrscheduler = LearningRateScheduler(scheduler)
        # storing history is optional, for plotting
        model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, lrscheduler])
    metric_values = model.evaluate(x=test_images, y=test_labels)
    print('Final TEST performance')
    for metric_value, metric_name in zip(metric_values, model.metrics_names):
        print('{}: {}'.format(metric_name, metric_value))
    return model, metric_values
