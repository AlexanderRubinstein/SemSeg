"""
ResNet based FCN.
"""
from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Convolution2D,
                          Conv2D,
                          Reshape,
                          Lambda,
                          # merge
                          Add )
import tensorflow as tf

from .resnet50 import ResNet50


FCN_RESNET = 'fcn_resnet'


def make_fcn_resnet(input_shape, nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None

    model = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor)

    if freeze_base:
        for layer in model.layers:
            layer.trainable = False

    x32 = model.get_layer('act3d').output
    x16 = model.get_layer('act4f').output
    x8 = model.get_layer('act5c').output

    # c32 = Convolution2D(nb_labels, 1, 1, name='conv_labels_32')(x32)
    # c16 = Convolution2D(nb_labels, 1, 1, name='conv_labels_16')(x16)
    # c8 = Convolution2D(nb_labels, 1, 1, name='conv_labels_8')(x8)

    c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
    c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
    c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)

    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    # m = merge([r32, r16, r8], mode='sum', name='merge_labels')

    # Change merge to Add() due to deprection of the former
    # m = Add()([r32, r16, r8], name='merge_add_labels')
    m = Add()([r32, r16, r8])

    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(input=input_tensor, output=x)

    return model
