import numpy as np
import tensorflow as tf

import vgg16

from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, Conv2D, UpSampling2D, Cropping2D

class InputNormalize(Layer):
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        return x/255.


def res_conv(nb_filter, nb_row, nb_col,stride=(1,1)):
    def _res_func(x):
        identity = Cropping2D(cropping=((2,2),(2,2)))(x)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(a)
        y = BatchNormalization()(a)
        return  add([identity, y])
    return _res_func


def conv_bn_relu(nb_filter, kr_row, kr_col, stride, relu=True):   
    def _conv_func(x):
        x = Conv2D(nb_filter, (kr_row, kr_col), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        if relu == True:
            x = Activation("relu")(x)
        else:
            x = Activation("tanh")(x)
        return x
    return _conv_func


"""
def dconv_bn_nolinear(nb_filter, nb_row, nb_col,stride=(2,2),activation="relu"):
    def _dconv_bn(x):
        x = Conv2DTranspose(nb_filter,nb_row, nb_col, output_shape=output_shape, subsample=stride, border_mode='same')(x)
        x = UpSampling2D(size=stride)(x)
        x = UnPooling2D(size=stride)(x)
        x = ReflectionPadding2D(padding=stride)(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    return _dconv_bn
"""
def dconv_bn_nolinear(nb_filter, kr_row, kr_col, stride=(2,2), activation="relu"):
    def _dconv_bn(x):
        x = Conv2DTranspose(nb_filter, (kr_row, kr_col), strides=stride, padding="same")(x)
        x = Activation(activation)(x)
        return x
    return _dconv_bn


class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''
    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''
        return (x + 1) * 127.5

    def compute_output_shape(self,input_shape):
        return input_shape


class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''
    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        x -= [123.68, 116.779, 103.93]
        #img_util.preprocess_image(style_image_path, img_width, img_height)
        return x

    def compute_output_shape(self,input_shape):
        return input_shape


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))
        self.input_spec = [InputSpec(ndim=4)] 

    def call(self, x, mask=None):
        top_pad=self.top_pad
        bottom_pad=self.bottom_pad
        left_pad=self.left_pad
        right_pad=self.right_pad
        paddings = [[0,0],[left_pad,right_pad],[top_pad,bottom_pad],[0,0]]
        return tf.pad(x,paddings, mode='REFLECT', name=None)

    def compute_output_shape(self,input_shape):
        rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
        cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None
        return (input_shape[0],
                rows,
                cols,
                input_shape[3])

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))     


class UnPooling2D(UpSampling2D):
    def __init__(self, size=(2, 2)):
        super(UnPooling2D, self).__init__(size) 

    def call(self, x, mask=None):
        shapes = x.get_shape().as_list() 
        w = self.size[0] * shapes[1]
        h = self.size[1] * shapes[2]
        return tf.image.resize(x, (w,h), method="nearest")


class InstanceNormalize(Layer):
    # 不知道这个是干嘛的
    def __init__(self, **kwargs):
        super(InstanceNormalize, self).__init__(**kwargs)
        self.epsilon = 1e-3

    def call(self, x, mask=None):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.truediv(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))                                             

    def compute_output_shape(self,input_shape):
        return input_shape
