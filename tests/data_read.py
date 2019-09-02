ls = ['math','os','sys','argparse','datetime','shutil','itertools','struct']
for i in ls:
    exec('import {0}'.format(i))
    #exec('print("imported {0}")'.format(i))

# Importing Standard Data Science & Deep Learning Libraries
ls = ['numpy','scipy','tensorflow','h5py','keras','sklearn','cv2','skimage']
for i in ls:
    exec('import {0}'.format(i))
    exec('print("Version of {0}",{0}.__version__)'.format(i))

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
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
# Advanced Activation Layers
advanced_activations = ('LeakyReLU','PReLU','ELU')
for i in advanced_activations:
    exec('from keras.layers import {}'.format(i))
# Normalization Layers
from keras.layers import BatchNormalization
# Vgg19 for Content loss applications
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
# Used for image.resize(target_size,PIL.Image.BICUBIC)
import PIL


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array




# Keras Backend for additional functionalities
from keras import backend as K

K.set_image_data_format('channels_last')

import numpy as np


float_precision = 'float32'
block_size = 96
scale = 4
overlap = 0.0
read_as_gray = False
bit_depth = 8

min_LR, max_LR =  0, 1
min_HR, max_HR = -1, 1

resize_interpolation = 'BICUBIC'

high_path = os.path.join('.','..','data','HR')
low_path  = os.path.join('.','..','data','LR')



######## DATA STORAGE, READING & PROCESSING FUNCTIONS BEGINS ########

datasets = {} # Global Dataset Storage
file_names = {} # Global filenames for disk_batching

def check_and_gen(name,low_path,high_path):
    "Checks for LR images in low_storage & create if doesn't exist"
    for ph in ('train','valid','test'):
        low_store  = os.path.join(low_path, name,ph)
        high_store = os.path.join(high_path,name,ph)
        if not os.path.isdir(low_store):
            os.makedirs(low_store)
        for file_name in os.listdir(high_store):
            if not os.path.isfile(os.path.join(low_store,file_name)):
                    clr = 'grayscale' if read_as_gray else 'rgb'
                    image = load_img(os.path.join(high_store,file_name),color_mode=clr)
                    image = image.resize( ((image.width//scale),(image.height//scale)) ,
                        resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                    image.save(os.path.join(low_store,file_name))

def on_fly_crop(mat,block_size=block_size,overlap=overlap):
    "Crop images into patches, with explicit overlap"
    ls = []
    nparts1, nparts2 = int(np.ceil(mat.shape[0]/block_size)), int(np.ceil(mat.shape[1]/block_size))
    step1, step2     = (mat.shape[0]-block_size)/(nparts1-1), (mat.shape[1]-block_size)/(nparts2-1)
    step1, step2     = step1 - int(np.ceil(step1*overlap))  , step2 - int(np.ceil(step2*overlap))
    for i in range(nparts1):
        i = round(i*step1)
        for j in range(nparts2):
            j = round(j*step2)
            ls.append( mat[i:i+block_size,j:j+block_size] )
    return ls
    
def feed_data(name, low_path, high_path, lw_indx, up_indx, crop_flag = True, phase = 'train', erase_prev=False):
    "feed the required fraction of dataset into memory for consumption by model"
    global datasets
    if erase_prev:
        datasets = {}
    check_and_gen(name,low_path,high_path)
    for x in ('LR','HR'):
        data_path = high_path if x=='HR' else low_path
        if x not in datasets:
            datasets[x] = {}
        if name not in datasets:
            datasets[x][name] = {}
        for ph in ('train','valid','test'):
            if ph not in datasets[x][name]:# All datasets be present, even if empty
                datasets[x][name][ph] = []
            if phase == ph:
                for img_name in file_names[phase][lw_indx:up_indx]:
                    clr = 'grayscale' if read_as_gray else 'rgb'
                    image = load_img(os.path.join(data_path,name,phase,img_name),color_mode=clr)
                    image = image.resize( ((image.width//scale)*scale,(image.height//scale)*scale) ,
                        resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                    # PIL stores image in WIDTH, HEIGHT shape format, Numpy as HEIGHT, WIDTH
                    img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
                    if crop_flag:
                        datasets[x][name][phase].extend(on_fly_crop(img_mat,block_size//(1 if x=='HR' else scale),overlap))
                    else:
                        datasets[x][name][phase].append(img_mat)
            #Converting into Numpy Arrays
            datasets[x][name][phase] = np.array(datasets[x][name][phase])

def normalize(mat,typ='HR'):
    "Normalize image matrix into ranges given"
    if str(mat.dtype)=='object':
        val = np.array([ (i.astype(float_precision)/((1<<bit_depth)-1)) for i in mat])
    else:
        val = mat.astype(float_precision) / ((1<<bit_depth)-1)
    return (eval('max_{}'.format(typ))-eval('min_{}'.format(typ)))*val + eval('min_{}'.format(typ))

def backconvert(mat,typ='HR'):
    mat = ( mat - eval('min_{}'.format(typ)) ) / (eval('max_{}'.format(typ))-eval('min_{}'.format(typ)))
    mat = mat*((1<<bit_depth)-1)
    return np.clip( mat.round(), 0, ((1<<bit_depth)-1) )

def get_data(name,phase,indx=['LR','HR'],org=False): # Flag to get as uint8 or float values
    res = []
    for x in indx:
        res.append( datasets[x][name][phase] )
        if not org:
            res[-1] = normalize(res[-1],x)
    return res if len(res)>1 else res[0]

######## DATA STORAGE, READING & PROCESSING FUNCTIONS ENDS ########



file_names['train'] = ['BaseArchitechture.png','Coffee-Mug.jpg']

feed_data('sample', low_path, high_path, None, None, crop_flag = True, phase = 'train', erase_prev=False)