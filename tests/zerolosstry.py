import keras
import numpy as np

from keras import backend as K


X = np.random.random((100,4,2))
Y = np.array([None]*100)

def zero_loss(y_true,y_pred):
	return K.mean(y_pred,axis=-1)

def model(shape):
	X0 = keras.layers.Input(shape)
	X1 = keras.layers.Dense(32)(X0)
	X2 = keras.layers.Dense(1)(X1)
	X3 = keras.layers.Concatenate()([X2,X2])
	return keras.models.Model(inputs=X0,outputs=X3)

mymodel = model((4,2))
opt = keras.optimizers.Adam(lr=0.1,decay=0.1)

mymodel.compile(optimizer=opt, loss=zero_loss)

mymodel.fit(X,Y,epochs=10)


