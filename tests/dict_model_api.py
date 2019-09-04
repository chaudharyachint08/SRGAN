import keras
import numpy as np

from keras import backend as K

samples = 100

X0 = np.random.random((samples,4,2))
X1 = np.random.random((samples,5,1))

Y0 = np.array( [ None   ]*samples )
Y1 = np.array( [ [[1]]*5, ]*samples )

def zero_loss(y_true,y_pred):
	return K.mean(y_pred,axis=-1)

def model(*shape):
	X0_0 = keras.layers.Input(shape[0],name='first_input')
	X0_1 = keras.layers.Input(shape[1],name='second_input')

	X1_0 = keras.layers.Dense(32)(X0_0)
	X1_1 = keras.layers.Dense(32)(X0_1)
	
	X2_0 = keras.layers.Dense(1,name='first_output' )(X1_0)
	X2_1 = keras.layers.Dense(1,name='second_output')(X1_1)

	return keras.models.Model(inputs=[X0_0,X0_1],outputs=[X2_0,X2_1])

mymodel = model((4,2),(5,1))


X    = { 'second_input':X1         , 'first_input':X0  }
Y    = { 'second_output':Y1        , 'first_output':Y0 }
loss = { 'second_output':'MSE'     , 'first_output':zero_loss }
loss_weights = { 'second_output':1 , 'first_output':1 }
opt = keras.optimizers.Adam()

mymodel.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)

mymodel.fit(X,Y)

res  = mymodel.predict(X)
res2 = mymodel.predict([X0,X1])

assert (res[0].shape==res2[0].shape) and (res[1].shape==res2[1].shape)

print(type(X),type(res))
print('First  I/O to Model',X0.shape,res[0].shape)
print('Second I/O to Model',X1.shape,res[1].shape)
mymodel.summary()


'''
 32/100 [========>.....................] - ETA: 1s - loss: 0.4302 - first_output_loss: -0.2208 - second_output_loss: 0.6
 100/100 [==============================] - 0s 5ms/step - loss: 0.4021 - first_output_loss: -0.2355 - second_output_loss: 0.6376
<class 'dict'> <class 'list'>
First  I/O to Model (100, 4, 2) (100, 4, 1)
Second I/O to Model (100, 5, 1) (100, 5, 1)
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
first_input (InputLayer)        (None, 4, 2)         0
__________________________________________________________________________________________________
second_input (InputLayer)       (None, 5, 1)         0
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4, 32)        96          first_input[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 5, 32)        64          second_input[0][0]
__________________________________________________________________________________________________
first_output (Dense)            (None, 4, 1)         33          dense_1[0][0]
__________________________________________________________________________________________________
second_output (Dense)           (None, 5, 1)         33          dense_2[0][0]
==================================================================================================
Total params: 226
Trainable params: 226
Non-trainable params: 0
__________________________________________________________________________________________________
>>>                                                                                                                     
'''