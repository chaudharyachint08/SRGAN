import keras
# Keras models, sequential & much general functional API
from keras.models import Sequential, Model
# Core layers
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Reshape, Lambda, SpatialDropout2D
# Convolutional Layers
from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose
from keras.layers import UpSampling2D, ZeroPadding2D
# Pooling Layers
from keras.layers import MaxPooling2D, AveragePooling2D
# Merge Layers
# from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
# Advanced Activation Layers
advanced_activations = ('LeakyReLU','PReLU','ELU')
for i in advanced_activations:
    exec('from keras.layers import {}'.format(i))
# Normalization Layers
from keras.layers import BatchNormalization
# Vgg19 for Content loss applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception

# from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
# keras image usage functions
# from keras.preprocessing.image import load_img, img_to_array


fun_params = list( globals().keys())
import inspect

dct = {}
for i in fun_params:
	try:
		_ = frozenset(inspect.signature(eval(i)).parameters.keys())
		dct[i] = _
		# print(i)
		# print(_)
	except:
		pass


dct2 = {}
for i in dct:
	if dct[i] not in dct2:
		dct2[ dct[i] ] = set()
	dct2[ dct[i] ].add( i )

for i in dct2:
	print(i)
	print(dct2[i])