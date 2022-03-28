# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:34:16 2018

@author: sehyung
"""

import sys
sys.path.insert(1, 'C:/Users/jo77pihe/Documents/MasterThesis_OfSpinesAndDendrites')

import tensorflow as tf
import tensorflow_addons as tfa

nf = 16
fz = 3
N = 0


def build_conv_layer(input_conv, num_filters=32, filter_sz=3, stride=1, padding='SAME', relu_op=False, norm=False,
                     name=''):
    if padding == 'VALID':
        if input_conv.shape[1] is not None:
            if (input_conv.shape[1] <=1):
                if not (input_conv.shape[2] <=1):
                    input_conv = tf.pad(input_conv, paddings=tf.constant([[0,0], [0,0], [1, 1], [1, 1], [0,0]]), mode="REFLECT")
            else:
                input_conv = tf.pad(input_conv, paddings=tf.constant([[0,0], [1, 1], [1, 1], [1, 1], [0,0]]), mode="REFLECT")
        else:
            input_conv = tf.pad(input_conv, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]),
                                mode="REFLECT")


        # input_conv = input_conv[1:-1,:,:,:,1:-1]
    global N
    nx = N
    N = nx+1
    name = name if name != '' else 'conv'+str(nx)
    if input_conv.shape[1] is not None and (input_conv.shape[1] <= 1):
        if (input_conv.shape[2] <= 1):
            filter_sz_x = (1,1,1)
        else:
            filter_sz_x = (1,filter_sz,filter_sz)
    else:
        filter_sz_x = filter_sz

    conv = tf.keras.layers.Conv3D(num_filters, filter_sz_x, stride, 'valid', activation=None, name=name)(input_conv)
    #conv = tf.keras.layers.Conv3D(num_filters, filter_sz, stride, 'same', activation=None, name=name)(input_conv)


    if relu_op:
        conv = tf.nn.leaky_relu(conv)
    if norm:
        conv = tfa.layers.InstanceNormalization()(conv)
    return conv


def build_upconv_layer(input_conv, num_filters=16, filter_sz=3, stride=(2, 2, 2), padding='SAME', relu_op=False,
                       norm=False, name=''):
    up_sample = tf.keras.layers.UpSampling3D(size=stride)(input_conv)
    if padding == 'VALID':
        if input_conv.shape[1] is not None and (up_sample.shape[1] <= 1):
            if not (up_sample.shape[2] <= 1):
                up_sample = tf.pad(up_sample, paddings=tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]),
                               mode="REFLECT")

        else:
            up_sample = tf.pad(up_sample, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]), mode="REFLECT")


    global N
    nx = N
    N = nx+1
    # conv = tf.keras.layers.Conv3D(num_filters, filter_sz, 1, 'valid', activation=None, name='conv'+str(nx))(up_sample)
    name = name if name != '' else 'conv'+str(nx)

    if up_sample.shape[1] is not None and up_sample.shape[1] <= filter_sz:
        if up_sample.shape[2] <= filter_sz:
            filter_sz_x = (1,1,1)
        else:
            filter_sz_x = (1,filter_sz,filter_sz)
    else:
        filter_sz_x = filter_sz
    conv = tf.keras.layers.Conv3D(num_filters, filter_sz_x, 1, 'valid', activation=None, name=name)(up_sample)

    if relu_op:
        conv = tf.nn.leaky_relu(conv)
    if norm:
        conv = tfa.layers.InstanceNormalization()(conv)

    return conv


def munet_cnn_level_0(L0_img, name='munet'):
    conv1 = build_conv_layer(L0_img, num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    if conv1.shape[1] == 1:
        if conv1.shape[2]==1:
            stride1 = (1,1,1)
        else:
            stride1=(1,2,2)
    else:
        stride1=(2,2,2)
    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    if conv2.shape[1] == 1:
        if conv2.shape[2]==1:
            stride2 = (1,1,1)
        else:
            stride2=(1,2,2)
    else:
        stride2=(2,2,2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)

    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID',
                                  relu_op=True, norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv3.shape[1] == 1:
        if conv3.shape[2]==1:
            stride3 = (1,1,1)
        else:
            stride3=(1,2,2)
    else:
        stride3=(2,2,2)
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
    L0_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID', name ='L0_pred')
    L0_L1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L0_pred)
    return L0_L1, L0_pred


def munet_cnn_level_1(L1_img, L0_L1= None, name = 'l1'):
    if L0_L1 is None:
        conv1 = build_conv_layer(tf.concat(L1_img, 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                                 relu_op=True, norm=False)
    else:
        conv1 = build_conv_layer(tf.concat([L1_img, L0_L1], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                                 relu_op=True, norm=False)
    if conv1.shape[1] == 1:
        if conv1.shape[2]==1:
            stride1 = (1,1,1)
        else:
            stride1=(1,2,2)
    else:
        stride1=(2,2,2)

    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv2.shape[1] == 1:
        if conv2.shape[2]==1:
            stride2 = (1,1,1)
        else:
            stride2=(1,2,2)
    else:
        stride2=(2,2,2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)

    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv3.shape[1] == 1:
        if conv3.shape[2]==1:
            stride3 = (1,1,1)
        else:
            stride3=(1,2,2)
    else:
        stride3=(2,2,2)
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
    L1_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID', name ='L1_pred')
    L1_L2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L1_pred)
    return L1_L2, L1_pred


def munet_cnn_level_2(L2_img, L1_L2= None, name = 'l2'):
    if L1_L2 is None:
        conv1 = build_conv_layer(tf.concat(L2_img, 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    else:
        conv1 = build_conv_layer(tf.concat([L2_img, L1_L2], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    if conv1.shape[1] == 1:
        if conv1.shape[2]==1:
            stride1 = (1,1,1)
        else:
            stride1=(1,2,2)
    else:
        stride1=(2,2,2)

    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv2.shape[1] == 1:
        if conv2.shape[2]==1:
            stride2 = (1,1,1)
        else:
            stride2=(1,2,2)
    else:
        stride2=(2,2,2)

    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)

    conv3_pool = build_conv_layer(conv3, num_filters=nf * 4, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv4 = build_conv_layer(conv3_pool, num_filters=nf * 8, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv3.shape[1] == 1:
        if conv3.shape[2]==1:
            stride3 = (1,1,1)
        else:
            stride3=(1,2,2)
    else:
        stride3=(2,2,2)
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
    L2_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID', name ='L2_pred')
    L2_L3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(L2_pred)
    return L2_L3, L2_pred


def munet_cnn_level_3(L3_img, L2_L3=None, name = 'l3'):
    if L2_L3 is None:
        conv1 = build_conv_layer(tf.concat(L3_img, 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    else:
        conv1 = build_conv_layer(tf.concat([L3_img, L2_L3], 4), num_filters=nf, filter_sz=fz, stride=1, padding='VALID',
                             relu_op=True, norm=False)
    if conv1.shape[1] == 1:
        if conv1.shape[2]==1:
            stride1 = (1,1,1)
        else:
            stride1=(1,2,2)
    else:
        stride1=(2,2,2)

    conv1_pool = build_conv_layer(conv1, num_filters=nf, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv2 = build_conv_layer(conv1_pool, num_filters=nf * 2, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv2.shape[1] == 1:
        if conv2.shape[2]==1:
            stride2 = (1,1,1)
        else:
            stride2=(1,2,2)
    else:
        stride2=(2,2,2)
    conv2_pool = build_conv_layer(conv2, num_filters=nf * 2, filter_sz=fz, stride=2, padding='VALID', relu_op=True,
                                  norm=True)

    conv3 = build_conv_layer(conv2_pool, num_filters=nf * 4, filter_sz=fz, stride=1, padding='VALID', relu_op=True,
                             norm=False)
    if conv3.shape[1] == 1:
        if conv3.shape[2]==1:
            stride3 = (1,1,1)
        else:
            stride3=(1,2,2)
    else:
        stride3=(2,2,2)

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
    L3_pred = build_conv_layer(conv7_1, num_filters=1, filter_sz=fz, stride=1, padding='VALID', name ='L3_pred')
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
