# Neural networks used in comparison
# U-net and FFN

from tensorflow.keras.layers import Input, Dropout, Conv2D, UpSampling2D, concatenate, AvgPool2D, Dense
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam


def unet_small(input_size=(4800, 6, 1)):
    inputs = Input(input_size)
    # Convolution 1
    kernelsize = (3, 6)  # 3->6 all 6 dyes
    kernelsize_up = (2, 6)  # one less because concatenated
    poolsize = (2, 1)  # pooling

    conv1 = Conv2D(2, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(inputs)
    conv1 = Conv2D(2, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv1)
    # Pooling 1
    pool1 = AvgPool2D(poolsize)(conv1)
    # Convolution 2
    conv2 = Conv2D(4, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(pool1)
    conv2 = Conv2D(4, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv2)
    # Pooling 2
    pool2 = AvgPool2D(poolsize)(conv2)
    # Convolution 3
    conv3 = Conv2D(8, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(pool2)
    conv3 = Conv2D(8, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv3)
    # Pooling 3
    pool3 = AvgPool2D(poolsize)(conv3)
    # Convolution 3
    conv4 = Conv2D(16, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(pool3)
    conv4 = Conv2D(16, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv4)
    # Dropout
    drop4 = Dropout(0.5)(conv4)

    # Upward Convolution 7
    up5 = Conv2D(8, kernelsize_up, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(poolsize)(drop4))
    # Here we copy the input from the upward convolution and contraction path
    merge5 = concatenate([conv3, up5])
    conv5 = Conv2D(8, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(merge5)
    conv5 = Conv2D(8, kernelsize, activation='relu', padding='same', kernel_initializer='he_normal'
                   )(conv5)
    # Upward Convolution 8
    up6 = Conv2D(4, kernelsize_up, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(poolsize)(conv5))
    # Here we copy the input from the upward convolution and contraction path
    merge6 = concatenate([conv2, up6])
    conv6 = Conv2D(4, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(merge6)
    conv6 = Conv2D(4, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv6)
    # Upward Convolution 9
    up7 = Conv2D(2, kernelsize, activation='relu', padding='same',
                 kernel_initializer='he_normal'
                 )(UpSampling2D(poolsize)(conv6))
    # Here we copy the input from the upward convolution and contraction path
    merge7 = concatenate([conv1, up7])
    conv7 = Conv2D(2, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(merge7)
    conv7 = Conv2D(2, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv7)
    conv7 = Conv2D(2, kernelsize, activation='relu', padding='same',
                   kernel_initializer='he_normal'
                   )(conv7)
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv8)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[BinaryAccuracy(), AUC()])

    return model


def FFN_DTDP(input_size=(1206,)):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=input_size))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # final layer should be softmax, but sigmoid is better for binary
    # categorical cross entropy loss is changed to binary
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[BinaryAccuracy(), AUC()])
    return model

def MHCNN_DT(filename = "data/EPG_classification_Keras_MHCNN_model.h5"):
    model = load_model(filename)
    return model

