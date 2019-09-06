import keras, tensorflow as tf
from keras.layers import Layer


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
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight( name='kernel' , shape=(len(input_shape), 1) , initializer='glorot_uniform' , trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
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