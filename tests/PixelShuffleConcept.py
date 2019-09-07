import keras, tensorflow as tf, numpy as np, keras.backend as K


block_size = 2
A = np.linspace(1,648,648).reshape((1,9,9,8))



print(A.shape)
# Same as Pixel Shuffle operation
B = K.eval( tf.depth_to_space(A,block_size=block_size) )
print(B.shape)


A, B = A[0], B[0]


t1 = B[0][:block_size].T[:block_size].T
t2 = A[0][0].T[:block_size**2].reshape((block_size,block_size))


print('We now know concept of Pixel Shuffle & DeShuffle (entire concept known as subPixel)',(t1==t1).all())