from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from PIL import Image
import PIL

import numpy as np

def npPSNR(y_true, y_pred):
    return (10.0 * np.log((((1<<bit_depth)-1) ** 2) / (np.mean(np.square(y_pred - y_true))))) / np.log(10.0)


image  = load_img('comic.png',color_mode='rgb')
org = img_to_array(image)
scale = 4

db = image.resize( ((image.width//scale),(image.height//scale)) ,  resample = PIL.Image.BICUBIC )
dn = image.resize( ((image.width//scale),(image.height//scale)) ,  resample = PIL.Image.NEAREST )

ub = db.resize( ((image.width),(image.height)) ,  resample = PIL.Image.BICUBIC )
un = dn.resize( ((image.width),(image.height)) ,  resample = PIL.Image.BICUBIC )




'''
image  = PIL.Image.open('comic.png')

'Approach 1 for finding PSNR value from BICUBIC interpolation (DownSample & Upsample respectively)'
image2 = image.resize( ((image.width//scale),(image.height//scale)) ,  resample = PIL.Image.BICUBIC )
image2 = image2.resize( ((image.width),(image.height)) ,  resample = PIL.Image.BICUBIC )
npPSNR(img_to_array(image),img_to_array(image2))
20.206261273576796

'Approach 2 for finding PSNR value from BICUBIC interpolation (DownSample & Upsample respectively)'
image2 = image.resize( ((image.width//scale),(image.height//scale)) ,  resample = PIL.Image.BICUBIC )
image2 = image2.resize( ((image.width//scale)*scale,(image.height//scale)*scale) ,  resample = PIL.Image.BICUBIC )
image = image.resize( ((image.width//scale)*scale,(image.height//scale)*scale) ,  resample = PIL.Image.BICUBIC )
npPSNR(img_to_array(image),img_to_array(image2))
20.710200966924365

'PSNR value given in SRGAN paper'
21.59
'''