# Imports for Custom Layers implementation
import keras, tensorflow as tf
from keras.layers import Layer
# Imports for Multi input data augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from copy import deepcopy
# Import for custom Upsampling Layers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend

class MyUpSampling2D(Layer):
    "Taken from TF source code for tf.keras.layers.UpSampling2D, with added bicubic"
    def __init__(self,size=(2, 2),data_format=None,interpolation='nearest',**kwargs):
        super(MyUpSampling2D, self).__init__(**kwargs)
        if type(size)!=tuple:
            size = (size,size)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        if interpolation not in {'nearest', 'bilinear','bicubic'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"`,'
            ' `"bilinear"` or `"bicubic"`.')
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        width  = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        return tensor_shape.TensorShape([input_shape[0], height, width, input_shape[3]])    # def call(self, inputs):
    def call(self, inputs):
        "This method is using Keras backend and would not support BiCubic UpSampling2D"
        input_shape = tensor_shape.TensorShape(input_shape).as_list() # CHECKPOINT< getting shape at runtime
        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        width  = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        if self.interpolation='nearest':
             return tf.image.resize_nearest_neighbor(inputs,(self.height, self.width),align_corners=True )
        elif self.interpolation='bilinear':
            return tf.image.resize_bilinear(         inputs,(self.height, self.width),align_corners=True )
        elif self.interpolation='bicubic':
            return tf.image.resize_bicubic(          inputs,(self.height, self.width),align_corners=True )
    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(MyUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PixelShuffle(Layer):
    "Merges Separate Convolutions into Neighborhoods, increasing image size"
    def __init__(self, size=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.supports_masking = True
        self.size = size
    def call(self, inputs):
        return tf.depth_to_space(inputs,block_size=self.size)
    def get_config(self):
        config = {'size': float(self.size)}
        base_config = super(PixelShuffle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        try:
            return (input_shape[0],)+(input_shape[1]*self.size,)+(input_shape[2]*self.size,)+(input_shape[3]//self.size**2,)
        except:
            return (input_shape[0],)+(None,)+(None,)+(input_shape[3]//self.size**2,)

class DePixelShuffle(Layer):
    "Creates Separate Convolutions from Neighborhoods, decreasing image size"
    def __init__(self, size=2, **kwargs):
        super(DePixelShuffle, self).__init__(**kwargs)
        self.supports_masking = True
        self.size = size
    def call(self, inputs):
        return tf.space_to_depth(inputs,block_size=self.size)
    def get_config(self):
        config = {'size': float(self.size)}
        base_config = super(DePixelShuffle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        try:
            return (input_shape[0],)+(input_shape[1]//self.size,)+(input_shape[2]//self.size,)+(input_shape[3]*self.size**2,)
        except:
            return (input_shape[0],)+(None,)+(None,)+(input_shape[3]//self.size**2,)

class WeightedSumLayer(Layer):
    "Weighted sum of multiple convolutions, behaviour similar to merge layers"
    def __init__(self, **kwargs):
        super(WeightedSumLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight( name='kernel' , shape=(len(input_shape), 1) , initializer='glorot_uniform' , trainable=True)
        super(WeightedSumLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, inputs):
        assert isinstance(inputs, list)
        inputs = tf.convert_to_tensor(inputs)
        inputs = K.transpose(inputs)
        output = K.dot(inputs, self.kernel)
        output = K.transpose(output)[0]
        return output
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


class MultiImageFlow(Sequence):
    "Data Augmentation Flow for Multi - IO keras models, with shared ImageDataGenerator"
    def __init__(self,datagen, X, Y, batch_size):
        # type of datagen should be 'unit' or either same as of X
        self.datagen = datagen
        if (type(X) in (list, tuple)):
            if (type(self.datagen) not in (list, tuple)):
                self.datagen = dict(enumerate( [deepcopy(self.datagen)]*len(X) ))
            else:
                self.datagen = dict(enumerate(self.datagen))
        elif (type(X) is dict):
            if (type(self.datagen) is not dict):
                self.datagen = { i:self.datagen for i in X }
        else:
            self.datagen_type = "unit"
            self.datagen = dict(enumerate( [self.datagen] ))

        for ele in ('X','Y'):
            exec( 'self.{0} = {0}'.format(ele) )
            if type(eval('self.{0}'.format(ele))) in (list, tuple, dict):
                exec( 'self.{0}_type = type({0})'.format(ele) )
                if type(eval('self.{0}'.format(ele)))!=dict: # convert to dictionary, if other iterable
                    exec( 'self.{0} = dict(enumerate(self.{0}))'.format(ele) )
            else:
                exec( 'self.{0}_type = "unit"'.format(ele) )
                exec( 'self.{0} = dict([(0,self.{0})])'.format(ele) )

        for ele in ('X','Y'):
            exec( 'self.{0}_gen_flow = dict()'.format(ele) )
            for key in eval('self.{0}'.format(ele)):
                x = self.X[key]                    if ele=='X' else self.X[list(self.X.keys())[0]]
                y = self.Y[list(self.Y.keys())[0]] if ele=='X' else self.Y[key]
                exec( 'self.{0}_gen_flow[{1}] = self.datagen[{2}].flow( x , y , batch_size=batch_size)'.format(
                    ele,repr(key),(repr(key) if ele=='X' else repr(list(self.datagen.keys())[0]))) )

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.X_gen_flow[ list(self.X_gen_flow.keys())[0] ].__len__()

    def __getitem__(self, index):
        global cself
        cself = self
        for ele in ('X','Y'):
            exec( 'self.{0}_batch = dict()'.format(ele) )
            for key in eval('self.{0}_gen_flow'.format(ele)):
                exec( 'self.{0}_batch[{1}] = self.{0}_gen_flow[{1}].__getitem__(index)[{2}]'.format(ele,repr(key),(0 if ele=='X' else 1)) )
            if eval('self.{0}_type'.format(ele))=="unit":
                exec( 'self.{0}_batch = self.{0}_batch[0]'.format(ele) )
            elif eval('self.{0}_type'.format(ele))==dict:
                exec( 'self.{0}_batch = dict([ (i,cself.{0}_batch[i]) for i in cself.{0}_batch ])'.format(ele) )
            else:
                exec( 'self.{0}_batch = [ cself.{0}_batch[i] for i in sorted(cself.{0}_batch) ]'.format(ele) )

        return self.X_batch, self.Y_batch


if __name__ == '__main__':
    # EDSR paper describes how a perceptual loss can be used without GAN for Super-Resolution
    # https://github.com/Golbstein/EDSR-Keras

    # Taking code for pixel shuffle from here
    # https://github.com/atriumlts/subpixel/issues/18

    # Another option if above link's code doesn't work is
    # https://github.com/atriumlts/subpixel/blob/master/subpixel.py

    # Sadly! both of codes doesn't work, TF has a better solution, so we write keras layers on TF backend only
    # https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space

    import numpy as np
    from keras.layers import Conv2D, Input
    def get_model(shape):
        X0 = Input( shape )
        X1 = Conv2D( 128,(3,3),padding='same' )(X0)
        X2 = PixelShuffle(   size=2 )(X1)
        X3 = DePixelShuffle( size=2 )(X2)
        model = keras.models.Model(inputs=X0,outputs=X3)
        return model

    model = get_model((128,128,3))
    X = np.random.random((1,128,128,3))
    Y = model.predict(X)