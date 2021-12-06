import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, \
    Activation, Concatenate


def conv_block(input, num_filters, snd_batch_normalization=True):
    initializer = RandomNormal(mean=0.0, stddev=0.01)
    x = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer=initializer,
               bias_initializer='zeros')(input)
    print(x.shape)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer=initializer,
               bias_initializer='zeros')(x)
    print(x.shape)
    if snd_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters, snd_batch_normalization=False)
    print(x.shape)
    p = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    print(p.shape)

    return x, p


def decoder_block(input, skip_features, num_filters):
    initializer = RandomNormal(mean=0.0, stddev=0.01)
    x = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), strides=(2, 2),
                        kernel_initializer=initializer, bias_initializer='zeros', padding='same')(input)
    print(x.shape)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, snd_batch_normalization=True)

    return x


def build_tracnet(input_shape):
    # entry point into graph of layers
    inputs = Input(shape=input_shape, batch_size=32)
    print(inputs.shape)

    # convolution and max-pooling operations with specified numbers of filters
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    # base
    b1 = conv_block(p3, 256)

    # convolution and upsampling operations with specified numbers of filters
    d1 = decoder_block(b1, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)

    # output
    initializer = RandomNormal(mean=0.0, stddev=0.01)
    outputs = Conv2D(filters=2, kernel_size=(3, 3), padding='same', kernel_initializer=initializer,
               bias_initializer='zeros')(d3)

    # build model
    model = Model(inputs, outputs, name='Tracnet')

    return model