'''
VGG original but kaffe link
http://www.robots.ox.ac.uk/~vgg/research/very_deep/


keras VGG usage link
https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

The VGG() class takes a few arguments that may only interest you if you are looking
to use the model in your own project, e.g. for transfer learning.

For example:

1. include_top (True): Whether or not to include the output layers for the model.
You don’t need these if you are fitting the model on your own problem.

2. weights (‘imagenet‘): What weights to load.You can specify None to not load
pre-trained weights if you are interested in training the model yourself from scratch.

3. input_tensor (None): A new input layer if you intend to fit the model on new data of a different size.

4. input_shape (None): The size of images that the model is expected to take
if you change the input layer.

5. pooling (None): The type of pooling to use when you are training a new set of output layers.

6. classes (1000): The number of classes (e.g. size of output vector) for the model.
'''

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19

# load the model
vgg_model = VGG19(weights='imagenet',include_top=False)


'''
>>> vgg_model.summary()
Model: "vgg19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0
=================================================================
Total params: 20,024,384
Trainable params: 20,024,384
Non-trainable params: 0
_________________________________________________________________
'''



# load an image from file
image = load_img('Coffee-Mug.jpg', target_size=(255,255),interpolation='bicubic')

# convert the image pixels to a numpy array
image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model, subtract mean of RGB
# Generator will produce [0,255]*3 valued images, VGG requires mean subtracted
image = preprocess_input(image)

# predict the probability across all output classes
# yhat = model.predict(image)

# convert the probabilities to class labels
# label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability
# label = label[0][0]

# print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))