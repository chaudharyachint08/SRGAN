'''
Below code provides example on
1. Usage of loss function without y_true
2. Dictionary based I/O on Model API does work
3. Dense layer works in keras on non-vector inputs

Note: Concatenate, Reshape, Dense(1) layers can be used for weighted sum layer,
but not sure if (None,None,3) based FCN network works
'''
import keras
import numpy as np

from keras import backend as K

samples = 1000

ip_shape1 = (4,2)
ip_shape2 = (5,3)

X0 = np.random.random( (samples,)+ip_shape1 )
X1 = np.random.random( (samples,)+ip_shape2 )

Y0 = np.array( [ None ]*samples )

# Erroneous code is corrected
# Y1 = np.ones( (samples,)+ip_shape2[:-1]+(ip_shape2[-1],) )
Y1 = np.ones( (samples,)+ip_shape2[:-1]+(1,) )

def zero_loss(y_true,y_pred):
	return K.mean(K.square(y_pred),axis=-1)

def model(*shape):
	X0_0 = keras.layers.Input(shape[0],name='ip0')
	X0_1 = keras.layers.Input(shape[1],name='ip1')

	X1_0 = keras.layers.Dense(32)(X0_0)
	X1_1 = keras.layers.Dense(32)(X0_1)
	
	X2_0 = keras.layers.Dense(1,name='op0' )(X1_0)
	X2_1 = keras.layers.Dense(1,name='op1')(X1_1)

	return keras.models.Model(inputs=[X0_0,X0_1],outputs=[X2_0,X2_1])

mymodel = model(ip_shape1,ip_shape2)


X            = { 'ip1':X1         , 'ip0':X0  }
Y            = { 'op1':Y1        , 'op0':Y0 }
loss         = { 'op1':'MSE'     , 'op0':zero_loss }
loss_weights = { 'op1':1 , 'op0':1 }
metrics      = {'op1':'mse'}

opt = keras.optimizers.Adam()

mymodel.compile(optimizer=opt, loss=loss, loss_weights=loss_weights, metrics=metrics)

mymodel.fit(X,Y,epochs=10)

print(end='\n\n')

res  = mymodel.predict(X       , verbose=True)
res2 = mymodel.predict([X0,X1] , verbose=True)

assert (res[0].shape==res2[0].shape) and (res[1].shape==res2[1].shape)

print(type(X),type(res))
print('First  I/O to Model',X0.shape,res[0].shape)
print('Second I/O to Model',X1.shape,res[1].shape)
mymodel.summary()


'''
Epoch 1/10
1000/1000 [==============================] - 0s 446us/step - loss: 0.2933 - op0_loss: 0.0031 - op1_loss: 0.2902 - op1_mean_squared_error: 0.2902
Epoch 2/10
1000/1000 [==============================] - 0s 47us/step - loss: 0.0646 - op0_loss: 2.2045e-04 - op1_loss: 0.0644 - op1_mean_squared_error: 0.0644
Epoch 3/10
1000/1000 [==============================] - 0s 51us/step - loss: 0.0524 - op0_loss: 3.3014e-05 - op1_loss: 0.0524 - op1_mean_squared_error: 0.0524
Epoch 4/10
1000/1000 [==============================] - 0s 45us/step - loss: 0.0446 - op0_loss: 3.3274e-06 - op1_loss: 0.0446 - op1_mean_squared_error: 0.0446
Epoch 5/10
1000/1000 [==============================] - 0s 42us/step - loss: 0.0379 - op0_loss: 1.6869e-07 - op1_loss: 0.0379 - op1_mean_squared_error: 0.0379
Epoch 6/10
1000/1000 [==============================] - 0s 49us/step - loss: 0.0317 - op0_loss: 2.3272e-09 - op1_loss: 0.0317 - op1_mean_squared_error: 0.0317
Epoch 7/10
1000/1000 [==============================] - 0s 46us/step - loss: 0.0260 - op0_loss: 8.1942e-11 - op1_loss: 0.0260 - op1_mean_squared_error: 0.0260
Epoch 8/10
1000/1000 [==============================] - 0s 47us/step - loss: 0.0207 - op0_loss: 5.7518e-12 - op1_loss: 0.0207 - op1_mean_squared_error: 0.0207
Epoch 9/10
1000/1000 [==============================] - 0s 51us/step - loss: 0.0159 - op0_loss: 2.4964e-14 - op1_loss: 0.0159 - op1_mean_squared_error: 0.0159
Epoch 10/10
1000/1000 [==============================] - 0s 46us/step - loss: 0.0118 - op0_loss: 6.7228e-15 - op1_loss: 0.0118 - op1_mean_squared_error: 0.0118


1000/1000 [==============================] - 0s 74us/step
1000/1000 [==============================] - 0s 20us/step
<class 'dict'> <class 'list'>
First  I/O to Model (1000, 4, 2) (1000, 4, 1)
Second I/O to Model (1000, 5, 3) (1000, 5, 1)
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
ip0 (InputLayer)                (None, 4, 2)         0
__________________________________________________________________________________________________
ip1 (InputLayer)                (None, 5, 3)         0
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4, 32)        96          ip0[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 5, 32)        128         ip1[0][0]
__________________________________________________________________________________________________
op0 (Dense)                     (None, 4, 1)         33          dense_1[0][0]
__________________________________________________________________________________________________
op1 (Dense)                     (None, 5, 1)         33          dense_2[0][0]
==================================================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
__________________________________________________________________________________________________
>>>       
'''