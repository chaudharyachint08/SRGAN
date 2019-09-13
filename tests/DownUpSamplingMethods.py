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



