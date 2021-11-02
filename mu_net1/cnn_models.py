# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:34:16 2018

@author: sehyung
"""

import tensorflow as tf

nf = 16
fz = 3
N = 0


def build_conv_layer(input_conv, num_filters=32, filter_sz=3, stride=1, padding='SAME', relu_op=False, norm=False):
    if padding == 'VALID':
        input_conv = tf.pad(input_conv, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
    global N
    nx = N
    N = nx+1
    conv = tf.keras.layers.Conv3D(num_filters, filter_sz, stride, 'valid', activation=None, name='conv'+str(nx))(input_conv)

    if relu_op:
        conv = tf.nn.leaky_relu(conv)
    if norm:
        # equals InstanceNorm (https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras)
        conv = tf.keras.layers.BatchNormalization(axis=[0, 1])(conv)
    return conv


def build_upconv_layer(input_conv, num_filters=16, filter_sz=3, stride=(2, 2, 2), padding='SAME', relu_op=False,
                       norm=False):
    up_sample = tf.keras.layers.UpSampling3D(size=stride)(input_conv)
    if padding == 'VALID':
        up_sample = tf.pad(up_sample, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    global N
    nx = N
    N = nx+1
    conv = tf.keras.layers.Conv3D(num_filters, filter_sz, 1, 'valid', activation=None, name='conv'+str(nx))(up_sample)

    if relu_op:
        conv = tf.nn.leaky_relu(conv)
    if norm:
        # equals InstanceNorm (https://stackoverflow.com/questions/68088889/how-to-add-instancenormalization-on-tensorflow-keras)
        conv = tf.keras.layers.BatchNormalization(axis=[0, 1])(conv)

    return conv


def munet_cnn_level_0(L0_img, name='munet'):
    conv1 = build_conv_layer(L0_img, num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride1 = (1, 2, 2) if conv1.shape[1] == 1 else (2, 2, 2)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride2 = (1, 2, 2) if conv2.shape[1] == 1 else (2, 2, 2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride3 = (1, 2, 2) if conv3.shape[1] == 1 else (2, 2, 2)
    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    conv4_up = build_upconv_layer(conv4, num_filters=nf * 4, filter_sz=fz, stride=stride3, padding='VALID',
                                  relu_op=True, norm=True)

    conv5 = build_conv_layer(tf.concat([conv4_up, conv3], 4), num_filters=nf * 4, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv5_up = build_upconv_layer(conv5, num_filters=nf * 2, filter_sz=fz, stride=stride2, padding='VALID',
                                  relu_op=True, norm=True)

    conv6 = build_conv_layer(tf.concat([conv5_up, conv2], 4), num_filters=nf * 2, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv6_up = build_upconv_layer(conv6, num_filters=nf * 1, filter_sz=fz, stride=stride1, padding='VALID',
                                  relu_op=True, norm=True)

    conv7 = build_conv_layer(tf.concat([conv6_up, conv1], 4), num_filters=nf * 1, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True)
    conv7_1 = build_conv_layer(conv7, num_filters=nf * 1, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    L0_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID')
    L0_L1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L0_pred)
    return L0_L1, L0_pred


def munet_cnn_level_1(L1_img, L0_L1, name = 'l1'):
    conv1 = build_conv_layer(tf.concat([L1_img, L0_L1], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride1 = (1, 2, 2) if conv1.shape[1] == 1 else (2, 2, 2)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride2 = (1, 2, 2) if conv2.shape[1] == 1 else (2, 2, 2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride3 = (1, 2, 2) if conv3.shape[1] == 1 else (2, 2, 2)
    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    conv4_up = build_upconv_layer(conv4, num_filters=nf * 4, filter_sz=fz, stride=stride3, padding='VALID',
                                  relu_op=True, norm=True)

    conv5 = build_conv_layer(tf.concat([conv4_up, conv3], 4), num_filters=nf * 4, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv5_up = build_upconv_layer(conv5, num_filters=nf * 2, filter_sz=fz, stride=stride2, padding='VALID',
                                  relu_op=True, norm=True)

    conv6 = build_conv_layer(tf.concat([conv5_up, conv2], 4), num_filters=nf * 2, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv6_up = build_upconv_layer(conv6, num_filters=nf * 1, filter_sz=fz, stride=stride1, padding='VALID',
                                  relu_op=True, norm=True)

    conv7 = build_conv_layer(tf.concat([conv6_up, conv1], 4), num_filters=nf * 1, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True)
    conv7_1 = build_conv_layer(conv7, num_filters=nf * 1, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    L1_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID')
    L1_L2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L1_pred)
    return L1_L2, L1_pred


def munet_cnn_level_2(L2_img, L1_L2, name = 'l2'):
    conv1 = build_conv_layer(tf.concat([L2_img, L1_L2], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride1 = (1, 2, 2) if conv1.shape[1] == 1 else (2, 2, 2)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride2 = (1, 2, 2) if conv2.shape[1] == 1 else (2, 2, 2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride3 = (1, 2, 2) if conv3.shape[1] == 1 else (2, 2, 2)
    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    conv4_up = build_upconv_layer(conv4, num_filters=nf * 4, filter_sz=fz, stride=stride3, padding='VALID',
                                  relu_op=True, norm=True)

    conv5 = build_conv_layer(tf.concat([conv4_up, conv3], 4), num_filters=nf * 4, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv5_up = build_upconv_layer(conv5, num_filters=nf * 2, filter_sz=fz, stride=stride2, padding='VALID',
                                  relu_op=True, norm=True)

    conv6 = build_conv_layer(tf.concat([conv5_up, conv2], 4), num_filters=nf * 2, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv6_up = build_upconv_layer(conv6, num_filters=nf * 1, filter_sz=fz, stride=stride1, padding='VALID',
                                  relu_op=True, norm=True)

    conv7 = build_conv_layer(tf.concat([conv6_up, conv1], 4), num_filters=nf * 1, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True)
    conv7_1 = build_conv_layer(conv7, num_filters=nf * 1, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    L2_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID', )
    L2_L3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L2_pred)
    return L2_L3, L2_pred


def munet_cnn_level_3(L3_img, L2_L3, name = 'l3'):
    conv1 = build_conv_layer(tf.concat([L3_img, L2_L3], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    stride1 = (1, 2, 2) if conv1.shape[1] == 1 else (2, 2, 2)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride2 = (1, 2, 2) if conv2.shape[1] == 1 else (2, 2, 2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    stride3 = (1, 2, 2) if conv3.shape[1] == 1 else (2, 2, 2)
    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    conv4_up = build_upconv_layer(conv4, num_filters=nf * 4, filter_sz=fz, stride=stride3, padding='VALID',
                                  relu_op=True, norm=True)

    conv5 = build_conv_layer(tf.concat([conv4_up, conv3], 4), num_filters=nf * 4, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv5_up = build_upconv_layer(conv5, num_filters=nf * 2, filter_sz=fz, stride=stride2, padding='VALID',
                                  relu_op=True, norm=True)

    conv6 = build_conv_layer(tf.concat([conv5_up, conv2], 4), num_filters=nf * 2, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True, norm=False)
    conv6_up = build_upconv_layer(conv6, num_filters=nf * 1, filter_sz=fz, stride=stride1, padding='VALID',
                                  relu_op=True, norm=True)

    conv7 = build_conv_layer(tf.concat([conv6_up, conv1], 4), num_filters=nf * 1, filter_sz=fz, stride=1,
                             padding='VALID', relu_op=True)
    conv7_1 = build_conv_layer(conv7, num_filters=nf * 1, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    L3_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID')
    return L3_pred


def discriminator(img, name='disc'):
    input_conv = build_conv_layer(img, num_filters=nf, filter_sz=fz, stride=1, padding='VALID', relu_op=True)

    conv1 = build_conv_layer(input_conv, num_filters=nf, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True)
    response = build_conv_layer(conv4, num_filters=1, filter_sz=1, stride=1, padding='VALID')

    return conv1, conv2, conv3, response
