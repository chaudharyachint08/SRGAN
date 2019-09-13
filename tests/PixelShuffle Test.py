import keras, tensorflow as tf
# Keras models, sequential & much general functional API
from keras.models import Sequential, Model
# Core layers
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Reshape, Lambda, SpatialDropout2D
# Convolutional Layers
from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose
from keras.layers import UpSampling2D, ZeroPadding2D

from keras.layers import BatchNormalization


import PIL
from PIL import Image
import os, gc, numpy as np
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


from myutils import PixelShuffle, DePixelShuffle, WeightedSumLayer, MultiImageFlow


dim,size,mult = 128,2,3

def mymodel(shape):
	X0 = Input(shape)
	# X = BatchNormalization()(X0)
	X = PixelShuffle(size=size)(X0)
	return Model(inputs=X0,outputs=X)

model = mymodel((dim,dim,mult*size**2))
# model = mymodel((None,None,mult*size**2))

X = np.array([np.array([ np.ones((dim,dim))*(i+1) for i in range(mult*size**2) ]).T])


Y = model.predict(X)

print(Y[0][:8,:8].T[0].T)