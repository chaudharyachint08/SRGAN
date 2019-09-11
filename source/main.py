######## STANDARD & THIRD PARTY LIBRARIES IMPORTS BEGINS ########

# Importing Standard Python Libraries
ls = ['math','os','sys','argparse','datetime','shutil','itertools','struct']
for i in ls:
    exec('import {0}'.format(i))
    #exec('print("imported {0}")'.format(i))

# Importing Standard Data Science & Deep Learning Libraries
ls = ['numpy','scipy','tensorflow','h5py','keras','sklearn','cv2','skimage']
for i in ls:
    exec('import {0}'.format(i))
    exec('print("Version of {0}",{0}.__version__)'.format(i))

import keras, tensorflow as tf
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
# keras image usage functions
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# Keras custom objects in which we will add custom losses and metrics
from keras.utils.generic_utils import get_custom_objects
# Keras Backend for additional functionalities
from keras import backend as K
K.set_image_data_format('channels_last')
# Used for image.resize(target_size,PIL.Image.BICUBIC)
import PIL
from PIL import Image
import os, gc, numpy as np
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import warnings
warnings.filterwarnings("ignore")

######## STANDARD & THIRD PARTY LIBRARIES IMPORTS ENDS ########


######## GLOBAL INITIAL VARIABLES BEGINS ########

parser = argparse.ArgumentParser()

# Bool Type Arguments
parser.add_argument("--prev_model"         , type=eval , dest='prev_model'         , default=False)
parser.add_argument("--train"              , type=eval , dest='train_flag'         , default=True)
parser.add_argument("--test"               , type=eval , dest='test_flag'          , default=True)
parser.add_argument("--data_augmentation"  , type=eval , dest='data_augmentation'  , default=True)
parser.add_argument("--imwrite"            , type=eval , dest='imwrite'            , default=True)
parser.add_argument("--read_as_gray"       , type=eval , dest='read_as_gray'       , default=False)
parser.add_argument("--gclip"              , type=eval , dest='gclip'              , default=False)
# Int Type Arguments
parser.add_argument("--scale"              , type=eval , dest='scale'              , default=4)
parser.add_argument("--patch_size"         , type=eval , dest='patch_size'         , default=96)
parser.add_argument("--patches_limit"      , type=eval , dest='patches_limit'      , default=None)
parser.add_argument("--min_LR"             , type=eval , dest='min_LR'             , default=0)
parser.add_argument("--max_LR"             , type=eval , dest='max_LR'             , default=1)
parser.add_argument("--min_HR"             , type=eval , dest='min_HR'             , default=-1)
parser.add_argument("--max_HR"             , type=eval , dest='max_HR'             , default=1)
parser.add_argument("--bit_depth"          , type=eval , dest='bit_depth'          , default=8)
parser.add_argument("--outer_epochs"       , type=eval , dest='outer_epochs'       , default=3)
parser.add_argument("--inner_epochs"       , type=eval , dest='inner_epochs'       , default=1)
parser.add_argument("--disk_batch"         , type=eval , dest='disk_batch'         , default=20)
parser.add_argument("--memory_batch"       , type=eval , dest='memory_batch'       , default=32)
parser.add_argument("--disk_batches_limit" , type=eval , dest='disk_batches_limit' , default=None)
parser.add_argument("--valid_images_limit" , type=eval , dest='valid_images_limit' , default=None)
parser.add_argument("--test_images_limit"  , type=eval , dest='test_images_limit'  , default=None)
parser.add_argument("--seed"               , type=eval , dest='np_seed'            , default=None)
parser.add_argument("--SUP"                , type=eval , dest='SUP'                , default=5)
parser.add_argument("--fSUP"               , type=eval , dest='fSUP'               , default=None)
parser.add_argument("--alpha"              , type=eval , dest='alpha'              , default=1)     # MS-SSIM specific
parser.add_argument("--beta"               , type=eval , dest='beta'               , default=1)     # MS-SSIM specific
parser.add_argument("--B"                  , type=eval , dest='B'                  , default=16)    # B blocks in Generator
parser.add_argument("--U"                  , type=eval , dest='U'                  , default=3)     # Recursive block's Units
# Float Type Arguments
parser.add_argument("--lr"                 , type=eval , dest='lr'                 , default=0.001) # Initial learning rate
parser.add_argument("--flr"                , type=eval , dest='flr'                , default=None)  # Final   learning rate
parser.add_argument("--decay"              , type=eval , dest='decay'              , default=0.0)   # Exponential Decay
parser.add_argument("--momentum"           , type=eval , dest='momentum'           , default=0.9)   # Momentum in case of SGD
parser.add_argument("--overlap"            , type=eval , dest='overlap'            , default=0.0)   # Explicit fraction of Overlap
parser.add_argument("--a"                  , type=eval , dest='a'                  , default=0.5)   # convexity b/w MS-SSIM & GL1
parser.add_argument("--fa"                 , type=eval , dest='fa'                 , default=None)  # convexity b/w MS-SSIM & GL1
parser.add_argument("--k1"                 , type=eval , dest='k1'                 , default=None)
parser.add_argument("--k2"                 , type=eval , dest='k2'                 , default=None)
parser.add_argument("--C1"                 , type=eval , dest='C1'                 , default=1)     # SSIM specific
parser.add_argument("--C2"                 , type=eval , dest='C2'                 , default=1)     # SSIM specific
parser.add_argument("--gnclip"             , type=eval , dest='gnclip'             , default=None)
parser.add_argument("--gvclip"             , type=eval , dest='gvclip'             , default=None)
parser.add_argument("--b"                  , type=eval , dest='b'                  , default=0.5)  # convexity b/w weighted & last
parser.add_argument("--fb"                 , type=eval , dest='fb'                 , default=None) # convexity b/w weighted & last
# String Type Arguments
parser.add_argument("--gen_choice"         , type=str , dest='gen_choice'          , default='baseline_gen')
parser.add_argument("--dis_choice"         , type=str , dest='dis_choice'          , default='baseline_dis')
parser.add_argument("--con_choice"         , type=str , dest='con_choice'          , default='baseline_con')
parser.add_argument("--attention"          , type=str , dest='attention'           , default='sigmoid')
parser.add_argument("--precision"          , type=str , dest='float_precision'     , default='float32')
parser.add_argument("--optimizer"          , type=str , dest='optimizer'           , default='Adam')
parser.add_argument("--name"               , type=str , dest='data_name'           , default='sample')
parser.add_argument("--train_strategy"     , type=str , dest='train_strategy'      , default='cnn') # other is 'gan'
parser.add_argument("--high_path"          , type=str , dest='high_path'           , default=os.path.join('.','..','data','HR'))
parser.add_argument("--low_path"           , type=str , dest='low_path'            , default=os.path.join('.','..','data','LR'))
parser.add_argument("--gen_path"           , type=str , dest='gen_path'            , default=os.path.join('.','..','data','SR'))
parser.add_argument("--save_dir"           , type=str , dest='save_dir'            , default=os.path.join('.','..','experiments','saved_models'))
parser.add_argument("--plots"              , type=str , dest='plots'               , default=os.path.join('.','..','experiments','training_plots'))
# parser.add_argument("--loss"             , type=str , dest='loss'                , default='MSE')
parser.add_argument("--resize_interpolation"   , type=str  , dest='resize_interpolation'   , default='BICUBIC')
parser.add_argument("--upsample_interpolation" , type=str  , dest='upsample_interpolation' , default='bilinear')
# Tuple Type Arguments
parser.add_argument("--channel_indx"           , type=eval , dest='channel_indx'           , default=(0,1,2))

args, unknown = parser.parse_known_args()
globals().update(args.__dict__)

######## GLOBAL INITIAL VARIABLES ENDS ########


######## CUSTOM IMPORTS AFTER COMMAND LINE ARGUMENTS PARSING BEGINS ########

from myutils import PixelShuffle, DePixelShuffle, WeightedSumLayer, MultiImageFlow
import models_collection
models_collection.initiate(globals())
from models_collection import configs_dict

######## CUSTOM IMPORTS AFTER COMMAND LINE ARGUMENTS PARSING ENDS ########


######## PROGRAM INITILIZATION BEGINS ########

np.random.seed(np_seed)
C1, C2 = k1**2 if (k1 is not None) else C1 , k2**2 if (k2 is not None) else C2
iSUP, ia, ib = SUP, a, b

######## PROGRAM INITILIZATION ENDS ########


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
        for img_name in os.listdir(high_store):
            if not os.path.isfile(os.path.join(low_store,img_name)):
                image = load_img(os.path.join(high_store,img_name),color_mode='rgb')
                image = image.resize( ((image.width//scale),(image.height//scale)) ,
                    resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                image.save(os.path.join(low_store,img_name))

def on_fly_crop(mat,patch_size=patch_size,overlap=overlap):
    "Crop images into patches, with explicit overlap, and limit of random patches if exist"
    ls = []
    nparts1, nparts2 = int(np.ceil(mat.shape[0]/patch_size)) , int(np.ceil(mat.shape[1]/patch_size))
    step1, step2     = (mat.shape[0]-patch_size)/(nparts1-1) , (mat.shape[1]-patch_size)/(nparts2-1)
    step1, step2     = step1 - int(np.ceil(step1*overlap))   , step2 - int(np.ceil(step2*overlap))
    for i in range(nparts1):
        i = round(i*step1)
        for j in range(nparts2):
            j = round(j*step2)
            ls.append( mat[i:i+patch_size,j:j+patch_size] )
    return ls
    
def feed_data(name, low_path, high_path, lw_indx, up_indx, crop_flag = True, phase = 'train', erase_prev=False):
    "feed the required fraction of dataset into memory for consumption by model"
    global datasets
    if erase_prev:
        datasets = {}
    check_and_gen(name,low_path,high_path)
    if patches_limit is None: # Faster in readling, as compared to ELSE part, due to multiple reshaping
        for x in ('LR','HR'):
            data_path = high_path if x=='HR' else low_path
            if x not in datasets:
                datasets[x] = {}
            if name not in datasets[x]:
                datasets[x][name] = {}
            for ph in ('train','valid','test'):
                # All datasets be present, even if empty
                if (ph not in datasets[x][name]):
                    datasets[x][name][ph] = []
                if phase == ph:
                    for img_name in file_names[name][ph][lw_indx:up_indx]:
                        clr = 'grayscale' if read_as_gray else 'rgb'
                        image = load_img(os.path.join(data_path,name,ph,img_name),color_mode=clr)
                        if x=='HR': # Squeeze & Stretch for HR images only
                            image = image.resize( ((image.width//scale)*scale,(image.height//scale)*scale) ,
                                resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                        # PIL stores image in WIDTH, HEIGHT shape format, Numpy as HEIGHT, WIDTH
                        img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
                        if crop_flag:
                            datasets[x][name][ph].extend(on_fly_crop(img_mat,patch_size//(1 if x=='HR' else scale) ))
                        else:
                            datasets[x][name][ph].append(img_mat)
                    datasets[x][name][ph] = np.array(datasets[x][name][ph])
    else:
        for x in ('LR','HR'):
            if x not in datasets:
                datasets[x] = {}
            if name not in datasets[x]:
                datasets[x][name] = {}
            for ph in ('train','valid','test'):
                # All datasets be present, even if empty
                if (ph not in datasets[x][name]):
                    datasets[x][name][ph] = []
        for img_name in file_names[name][phase][lw_indx:up_indx]:
            clr = 'grayscale' if read_as_gray else 'rgb'
            image = load_img(os.path.join(high_path,name,phase,img_name),color_mode=clr)
            # PIL stores image in WIDTH, HEIGHT shape format, Numpy as HEIGHT, WIDTH
            if crop_flag:
                image = image.resize( ((image.width//scale)*scale,(image.height//scale)*scale) ,
                    resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
                patch_ls = on_fly_crop(img_mat,patch_size,overlap)[:patches_limit]
                np.random.shuffle(patch_ls)
                patch_ls = patch_ls[:patches_limit]
                for x,scl in zip(('HR','LR'),(scale,1)):
                    if x=='LR':
                        for mat in patch_ls:
                            image = Image.fromarray(mat)
                            image = image.resize( ((image.width//scale),(image.height//scale)) ,
                                resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                            img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
                            datasets[x][name][phase].append(img_mat)
                    else:
                        datasets[x][name][phase].extend(patch_ls)
            # else:
            #     for x,scl in zip(('HR','LR'),(scale,1)):
            #         image = image.resize( ((image.width//scale)*scl,(image.height//scale)*scl) ,
            #             resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
            #         img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
            #         datasets[x][name][phase].append(img_mat)
        for x in ('LR','HR'):
            for ph in ('train','valid','test'):
                if ph==phase:
                    datasets[x][name][ph] = np.array(datasets[x][name][ph])

def normalize(mat,typ='HR'):
    "Normalize image matrix into ranges given"
    if str(mat.dtype)=='object':
        val = np.array([ ( i.astype(float_precision) / ((1<<bit_depth)-1) ) for i in mat])
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

from IPython.display import display
def show_ix(mat,ix,preproc=True,typ='HR'):
    "Function to Show images, for debugging purpose"
    if preproc:
        img = Image.fromarray(  backconvert(mat[ix],typ).astype('uint8')  )
    else:
        img = Image.fromarray(  mat[ix]  )
    display(img)
######## DATA STORAGE, READING & PROCESSING FUNCTIONS ENDS ########


######## METRIC DEFINITIONS BEGINS ########

def PSNR(y_true, y_pred, typ='HR'):
    y_true = ( y_true - eval('min_{}'.format(typ)) ) / (eval('max_{}'.format(typ))-eval('min_{}'.format(typ)))
    y_pred = ( y_pred - eval('min_{}'.format(typ)) ) / (eval('max_{}'.format(typ))-eval('min_{}'.format(typ)))
    y_true, y_pred = y_true*((1<<bit_depth)-1), y_pred*((1<<bit_depth)-1)
    y_true, y_pred = K.clip( K.round(y_true), 0, ((1<<bit_depth)-1) ), K.clip( K.round(y_pred), 0, ((1<<bit_depth)-1) )
    return (10.0 * K.log((((1<<bit_depth)-1) ** 2) / (K.mean(K.square(y_pred - y_true))))) / K.log(10.0)

def npPSNR(y_true, y_pred):
    return (10.0 * np.log((((1<<bit_depth)-1) ** 2) / (np.mean(np.square(y_pred - y_true))))) / np.log(10.0)

def get_IPSNR(true_tensor,pred_tensor,pred_mode='HR'):
    "return average PSNR between two tensors containing Images"
    avg_PSNR = [0,0]
    if (pred_mode=='HR') or (pred_mode=='LR' and bit_depth==8):
        for i in range(len(true_tensor)):
            true_img = backconvert(true_tensor[i] , typ='HR' )
            pred_img = backconvert(pred_tensor[i] , typ=pred_mode )
            # PIL supports only 8 bit wide images for now, makes sense so restrict initial PSNR
            if pred_mode=='LR':
                pred_img = pred_img.astype('uint8')
                pred_img = Image.fromarray(pred_img)
                pred_img = pred_img.resize( ((pred_img.width*scale),(pred_img.height*scale)) ,
                    resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                pred_img = img_to_array(pred_img)
            val = npPSNR(true_img,pred_img)
            if val!=float('inf'):
                avg_PSNR[0] += val
                avg_PSNR[1] += 1
    return (avg_PSNR[0]/(avg_PSNR[1]) if avg_PSNR[1] else avg_PSNR[0])

######## METRIC DEFINITIONS ENDS ########


######## CUSTOM LOSS FUNCTIONS DEFINITIONS BEGINS ########

def fgauss(size,sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def Kfgauss(size,sigma,channels):
    fg = np.array([fgauss(size,sigma)]*channels).T[:,:,:,np.newaxis]
    return K.constant( fg )

def MIX_LOSS( X,Y,a=ia,M=iSUP, C1=C1,C2=C2, alpha=alpha,beta=beta ):
    channels = len(channel_indx)
    Wo = K.ones((M,M,channels,1)) / M**2

    def l(scale):
        Wg = Kfgauss(M,scale,channels)
        UX, UY = K.depthwise_conv2d(X,Wg,padding='same'), K.depthwise_conv2d(Y,Wg,padding='same')
        return (2*UX*UY+C1)/(K.square(UX)+K.square(UY)+C1)

    def cs(scale):
        Wg = Kfgauss(M,scale,channels)
        UX, UY = K.depthwise_conv2d(X,Wg,padding='same'), K.depthwise_conv2d(Y,Wg,padding='same')
        SDX = K.depthwise_conv2d(K.square(X) ,Wg,padding='same') - K.square(UX)
        SDY = K.depthwise_conv2d(K.square(Y) ,Wg,padding='same') - K.square(UY)
        CXY = K.depthwise_conv2d( X*Y        ,Wg,padding='same') -        UX*UY
        return (2*CXY+C2)/(SDX+SDY+C2)

    def L1(scale):
        Wg = Kfgauss(M,scale,channels)
        return K.abs(X-Y)*K.max(Wg)
    MS_SSIM = l(M)**alpha
    #min_scale, max_scale = 0.1,M
    #r = (max_scale/min_scale)**(1/(M-1))
    for i in range(1,M+1):
        #i = min_scale * r**(i-1)
        MS_SSIM *= (cs(i)**beta)
    GL1  = L1(M)
    return a * K.mean((1-MS_SSIM), axis=-1)  +  (1-a) * K.mean(GL1, axis=-1)

def ZERO_LOSS(y_true,y_pred):
    return K.mean(y_pred,axis=-1)

def DIS_LOSS(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)

def ADV_LOSS(y_true, y_pred):
    y_pred = 1-y_pred
    return K.binary_crossentropy(y_true, y_pred)

######## CUSTOM LOSS FUNCTIONS DEFINITIONS ENDS ########


######## KERAS FUNCTIONS UPDATE BEGINS ########

keras_update_dict = {}
for l in ('MIX_LOSS','ZERO_LOSS','DIS_LOSS','ADV_LOSS',):
    keras_update_dict[l] = eval(l)
for loss in ('WeightedSumLayer','PixelShuffle','DePixelShuffle'):
    keras_update_dict[l] = eval(l)
keras_update_dict['PSNR'] = eval('PSNR')

get_custom_objects().update(keras_update_dict)

######## KERAS FUNCTIONS UPDATE ENDS ########


######## MODELS DEFINITIONS BEGINS ########

def dict_to_model_parse(config,*ip_shapes):
    "Model parser, these names are to be used in defining dictionaries"
    all_lyr_typs = ('clpre','dense','actvn','drpot','flttn','input','reshp','lmbda','spdrp',
        'convo','usmpl','zpdng','poolg','merge','aactv','btnrm','wsuml','pslyr','block',)
    for lyr_typ in all_lyr_typs:
        if lyr_typ != 'clpre':
            config[lyr_typ] = [ tuple(sorted(i)) for i in config[lyr_typ]  ] if lyr_typ in config else []
    if 'clpre' not in config:
        config['clpre'] = []
    if config['clpre']: # if ContentLayersModel is defined, then no other architecture is there to setup
        ConModel = eval( 'keras.applications.{}.{}'.format(''.join(config['clpre'][0]),config['clpre_sub'][0]) )
        con_model = ConModel( include_top=False, weights='imagenet', input_shape=(patch_size,patch_size,len(channel_indx)) )
        if 'clpre_out' in config: # if defined after which convo layer to take content, else last convolution layer is taken
            model = Model( inputs = con_model.input , outputs = con_model.get_layer(config['clpre_out'][0]).output )
        else:
            model = con_model
        if ('clpre_act' in config) and (not config['clpre_act'][0]):
                model.layers[-1].activation = None
    else: # Setting up architecture for GNERATOR/DISCRIMINATOR
        target, tensors = len(ip_shapes), [ Input(i) for i in ip_shapes ]
        while True:
            for lyr_typ in all_lyr_typs:
                ls = [ i[-1] for i in config[lyr_typ]  ] if lyr_typ in config else []
                if target in ls: # Which layer type is to be constructed and its index in category
                    cat_ix = ls.index(target)
                    break
            else: # if for loop iterates on all_lyr_typs, still doesn't break, means this target is not in config
                break
            try:
                lyr_args = config[lyr_typ+'_par'][cat_ix] # Collecting argumenst to pass while construction layer
            except:
                lyr_args = {}
            # Core layers are only individual in theri category, so no mention can be used directly
            core_lyrs = { 'dense':'Dense', 'actvn':'Activation', 'drpot':'Dropout', 'flttn':'Flatten',
                'input':'Input', 'reshp':'Reshape', 'lmbda':'Lambda', 'spdrp':'SpatialDropout2D' }
            multi_input_lyr, nxt_lyr = False, None # if layer has multiple inputs (MERGE/BLOCK)
            if   lyr_typ in core_lyrs:
                lyr_spec = core_lyrs[lyr_typ]            
            elif lyr_typ in ('convo','usmpl','poolg'):
                if lyr_typ == 'convo':
                    lyr_spec = config[lyr_typ+'_sub'][cat_ix]
                else:
                    lyr_spec = 'UpSampling2D' if lyr_typ == 'usmpl' else 'ZeroPadding2D'
            elif lyr_typ=='poolg':
                lyr_spec = config[lyr_typ+'_sub'][cat_ix]
            elif lyr_typ=='merge':
                lyr_spec = config[lyr_typ+'_sub'][cat_ix]
                multi_input_lyr = True
            elif lyr_typ=='aactv':
                lyr_spec = config[lyr_typ+'_sub'][cat_ix]
            elif lyr_typ=='btnrm':
                lyr_spec = 'BatchNormalization'
            elif lyr_typ=='wsuml':
                nxt_lyr  = WeightedSumLayer
                multi_input_lyr = True
            elif lyr_typ=='pslyr': # lyr_spec is 'PixelShuffle'/'DePixelShuffle'
                lyr_spec = config[lyr_typ+'_sub'][cat_ix]
                nxt_lyr  = eval( lyr_spec )
            elif lyr_typ=='block':
                lyr_spec = config[lyr_typ+'_sub'][cat_ix]
                nxt_lyr  = eval( 'configs_dict[{}]'.format(repr(lyr_spec)) )
                multi_input_lyr = True
            if nxt_lyr is None:
                nxt_lyr  = eval( '{}({})'.format( lyr_spec , ','.join('{}={}'.format(x,lyr_args[x]) for x in lyr_args) ) )
            if multi_input_lyr:
                tensors.append(  nxt_lyr( [ tensors[x] for x in config[lyr_typ][cat_ix][:-1] ] ) )
            else:
                tensors.append(  nxt_lyr( tensors[ config[lyr_typ][cat_ix][0] ] ) )
            target += 1
        model = Model( inputs = tensors[:len(ip_shapes)] , outputs = tensors[-1] )
    model.name = config['name']
    return model

def my_gan(*shapes):
    global generator_model, discriminator_model, content_model
    if len(shapes)==1:  shapes = shapes+shapes
    if len(shapes)==2:  shapes = shapes+(len(channel_indx),)
    generator_model     = dict_to_model_parse( configs_dict[gen_choice],(shapes[0]//scale , shapes[1]//scale , shapes[2]) )
    discriminator_model = dict_to_model_parse( configs_dict[dis_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    content_model       = dict_to_model_parse( configs_dict[con_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    generator_model.name = 'generator'
    X_lr   = Input((shapes[0]//scale , shapes[1]//scale , shapes[2]),name='lr_input')
    X_hr   = Input((shapes[0]        , shapes[1]        , shapes[2]),name='hr_input')
    Y_sr   = generator_model(X_lr)
    # Actual is given 1, fake is given 0
    Y_dis_sr  = discriminator_model(Y_sr)
    Y_dis_hr  = discriminator_model(X_hr)
    Y_dis     = Concatenate(name='discriminator')([Y_dis_sr,Y_dis_hr])
    # Sum of Square of content layers, mean is by ZERO_LOSS
    con_hr = content_model(X_hr)
    con_sr = content_model(Y_sr)
    Y_con  = Subtract()([con_sr,con_hr])
    Y_con  = Lambda( lambda x : K.square(x), name='content' ) (Y_con)
    return Model(inputs=[X_lr,X_hr],outputs=[Y_sr,Y_dis,Y_con],name=', '.join((gen_choice,dis_choice,con_choice)))

def freeze_model(model,recursive=False):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, Model) and recursive:
            freeze_model(layer,recursive)

def unfreeze_model(model,recursive=False):
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True
        if isinstance(layer, Model) and recursive:
            unfreeze_model(layer,recursive)

def my_model_save(model,name):
    with open("{}.json".format(name), "w") as json_file:
        json_file.write( model.to_json() )
    model.save_weights("{}.hdf5".format(name))

def my_model_load(model,name):
    model.load_weights('{}.hdf5'.format(name))

def compile_model(model,mode,opt):
    if mode in ('cnn','gen'):
        train_model , non_train_model = generator_model, discriminator_model
        loss = { 'generator':'MSE' , 'discriminator':'ADV_LOSS' , 'content':'ZERO_LOSS' }
        if mode=='cnn':
            loss_weights = { 'generator':1 , 'discriminator':0 , 'content':0 }
        elif mode=='gen':
            loss_weights = { 'generator':0 , 'discriminator':1e-3 , 'content':(1/12.75)**2 }
    elif mode=='dis':
        non_train_model , train_model = generator_model, discriminator_model
        loss = { 'generator':'MSE' , 'discriminator':'DIS_LOSS' , 'content':'ZERO_LOSS' }
        loss_weights = { 'generator':0 , 'discriminator':1 , 'content':0 }
    metrics = {'generator':'PSNR'}
    freeze_model(   content_model   )
    freeze_model(   non_train_model )
    unfreeze_model( train_model     )
    for layer in model.layers:
        if isinstance(layer, Model):
            if layer.name == 'content':
                layer.trainable = False
            elif layer.name == 'generator':
                layer.trainable == True  if mode in ('cnn','gen') else False
            elif layer.name == 'discriminator':
                layer.trainable = False if mode in ('cnn','gen') else True

    model.compile(  optimizer=opt , loss = loss , loss_weights = loss_weights, metrics=metrics )
    # freeze_model(   content_model   )
    # freeze_model(   non_train_model )
    # unfreeze_model( train_model     )

######## MODELS DEFINITIONS ENDS ########


######## PLOTTING FUNCTIONS BEGINS HERE ########

def plot_history(history):
    if train_strategy == 'cnn':
        plot_dir =  os.path.join(plots,gen_choice)
    else:
        plot_dir =  os.path.join(plots,gan_model.name)
    if not os.path.isdir(plot_dir):
        try:
            if os.path.isdir(plot_dir):
                shutil.rmtree(plot_dir)
            os.makedirs(plot_dir)
        except:
            pass
    for key in history.keys():
        if not key.startswith('val'):
            _ = plt.plot(history[key]         , linewidth=1 , label=key )
            _ = plt.plot( history['val_'+key] , linewidth=1 , label='val_'+key )
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3g}')) # 2 decimal places
            plt.xticks( np.round(np.linspace(0,len(history[key])-1,5),0) , np.round(np.linspace(1,len(history[key]),5),0) )
            plt.grid(True) ; plt.xlabel('Epochs')
            plt.ylabel(key) ; plt.title(' '.join(x for x in key.upper().split('_')))
            plt.legend( loc='upper left' , bbox_to_anchor=(1,1) , fancybox=True , shadow=True )
            plt.savefig( os.path.join(plot_dir,'{}.PNG'.format(key)) , dpi=600 , bbox_inches='tight' , format='PNG' )
            plt.close()

######## PLOTTING FUNCTIONS ENDS HERE ########


######## LEARNING RATE SCHEDULES BEGINS HERE ########

# Presently Arihtmetic Progression based between [lf,flr], O_E*N_DB
pass

######## LEARNING RATE SCHEDULES BEGINS HERE ########


######## TRAINING CODE SECTION BEGINS ########

def train(name,train_strategy,dis_gen_ratio=(1,1)):
    ""
    global iSUP, ia, ib, gan_model, all_history, history
    iSUP, ia, ib = SUP, a, b
    try:
        del datasets['LR'][name], datasets['HR'][name]
    except:
        pass
    # Modes in which have to train model, either CNN only interleaving using GAN
    modes = ['cnn'] if train_strategy=='cnn' else ['dis']*dis_gen_ratio[0]+['gen']*dis_gen_ratio[1]
    # Storing file names for retrieval of images
    if name not in file_names:
        file_names[name] = {}
    file_names[name]['train'] = sorted(os.listdir(os.path.join(high_path,name,'train')))
    file_names[name]['valid'] = sorted(os.listdir(os.path.join(high_path,name,'valid')))
    # Finding number of diak batches, also printing for user
    n_disk_batches = disk_batches_limit if (disk_batches_limit is not None) else int(np.ceil(len(file_names[name]['train'])/disk_batch))
    print('\nOuter Epochs {} Disk Batches {}'.format(outer_epochs,n_disk_batches))
    # Arithmetic progression based decay of learning rate, support window for MIX, convexity & weighted priority
    lr_delta  = 0 if (flr  is None) else ((flr-lr)   / (n_disk_batches*outer_epochs-1))
    SUP_delta = 0 if (fSUP is None) else ((fSUP-SUP) / (n_disk_batches*outer_epochs-1))
    a_delta   = 0 if (fa   is None) else ((fa-a)     / (n_disk_batches*outer_epochs-1))
    b_delta   = 0 if (fb   is None) else ((fb-b)     / (n_disk_batches*outer_epochs-1))
    # Reading Validation Set & finding initial PSNR difference from HR
    np.random.shuffle(file_names[name]['valid'])
    print('\nReading Validation Set',end=' ') ; init_time = datetime.now()
    feed_data(name,low_path,high_path,None,valid_images_limit,True,'valid')
    print( 'Time taken is {} '.format((datetime.now()-init_time)) )
    x_valid = get_data(name,'valid',indx=['LR','HR'],org=False)
    x_valid = {
    'lr_input': x_valid[0] ,
    'hr_input': x_valid[1]
    }
    y_valid = {
    'generator'    :x_valid['hr_input'] ,
    'discriminator':np.array([[0,1],]*len(x_valid['lr_input']),dtype=float_precision) ,
    'content'      :np.array([None]  *len(x_valid['lr_input']))
    }
    val_init_PSNR = get_IPSNR(x_valid['hr_input'],x_valid['lr_input'],pred_mode='LR')
    if train_strategy=='cnn':
        x_valid, y_valid = x_valid['lr_input'], x_valid['hr_input']
    # Selecting optimizer for training
    if optimizer == 'Adam':
        opt = keras.optimizers.Adam( lr=lr, decay=decay )
    elif optimizer == 'SGD':
        opt = keras.optimizers.SGD( lr=lr, decay=decay, momentum=momentum )
    else:   
        raise Exception('Unexpected Optimizer than two classics')
    # collecting history from all opechs, and oter_epochs training loop
    all_history = None
    for epc in range(outer_epochs):
        print('\nOuter Epoch {}'.format(epc+1))
        for mode in modes:
            if train_strategy!='cnn':
               print( 'Executing GAN in {} mode'.format('GENERATOR' if mode=='gen' else 'DISCRIMINATOR') )
            if train_strategy=='gan':
                model = gan_model
                compile_model(model,        mode , opt)
            else:
                model = generator_model
                model.compile(optimizer=opt,loss='MSE',metrics=['PSNR'])

            if (prev_model or epc) and os.path.isdir(save_dir): # after 1st outer epoch, due to recompiing models
                my_model_load( generator_model     , os.path.join(save_dir,gen_choice) )
                if train_strategy=='gan':
                    my_model_load( discriminator_model , os.path.join(save_dir,dis_choice) )
                    my_model_load( content_model       , os.path.join(save_dir,con_choice) )
            np.random.shuffle(file_names[name]['train'])
            iSUP = int(np.round(iSUP)) if int(np.round(iSUP))%2 else int(np.round(iSUP))+1
            
            for i in range(n_disk_batches):
                gc.collect()
                K.set_value(model.optimizer.lr    , lr+(i+epc*n_disk_batches)*lr_delta)
                K.set_value(model.optimizer.decay , decay)
                if gclip:
                    if gnclip is not None:
                        model.optimizer.__dict__['clipnorm']  = gnclip / lr+(i+epc*n_disk_batches)*lr_delta
                    if gvclip is not None:
                        model.optimizer.__dict__['clipvalue'] = gvclip / lr+(i+epc*n_disk_batches)*lr_delta
                
                print( 'Reading Disk Batch {}'.format(i+1) )
                init_time = datetime.now()
                feed_data(name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,True,'train')
                print( 'Time taken is {} '.format((datetime.now()-init_time)) )

                x_train = get_data(name,'train',indx=['LR','HR'],org=False)
                x_train = {
                'lr_input': x_train[0] ,
                'hr_input': x_train[1]
                }
                y_train = {
                'generator'    :x_train['hr_input'] ,
                'discriminator':np.array([[0,1],]*len(x_train['lr_input']),dtype=float_precision) ,
                'content'      :np.array([None]  *len(x_train['lr_input']))
                }
                train_init_PSNR = get_IPSNR(x_train['hr_input'],x_train['lr_input'],pred_mode='LR')
                if train_strategy=='cnn':
                    x_train, y_train = x_train['lr_input'], x_train['hr_input']
                if data_augmentation:
                    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
                    # datagen.fit(x_train) # this is not given in any GitHUb discussion or StackOverflow in README.md
                    if train_strategy=='cnn': # flow for single inpute keras model
                        flow = datagen.flow(x_train, y_train,batch_size=memory_batch)
                    else:
                        flow = MultiImageFlow(datagen,x_train,y_train,batch_size=memory_batch)
                    history = model.fit_generator(flow,epochs=inner_epochs,validation_data=(x_valid, y_valid))
                else:
                    history = model.fit(x=x_train,y=y_train,validation_data=(x_valid,y_valid),epochs=inner_epochs,batch_size=memory_batch)
                if train_strategy=='gan':                    
                    history = history.history
                    history['generator_IPSNR']     = [ (i-train_init_PSNR) for i in history['generator_PSNR'     ] ]
                    history['val_generator_IPSNR'] = [ (i-val_init_PSNR)   for i in history['val_generator_PSNR'] ]
                else:
                    history = history.history
                    history['IPSNR']     = [ (i-train_init_PSNR) for i in history['PSNR'    ] ]
                    history['val_IPSNR'] = [ (i-val_init_PSNR)   for i in history['val_PSNR'] ]
                del x_train, y_train
                all_history = history if (all_history is None) else { i:(all_history[i]+history[i])for i in history }
                plot_history(all_history) # Plotting progress after each disk batch
                #Saving Trained model, after all QP values of currect disk_batch were processed
                print( '\nSaving Trained model > ',end='' )
                # Save model and weights
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                try:
                    my_model_save( generator_model         , os.path.join(save_dir,gen_choice ) )
                    if train_strategy=='gan':
                        my_model_save( discriminator_model , os.path.join(save_dir,dis_choice ) )
                        my_model_save( content_model       , os.path.join(save_dir,con_choice ) )
                        # freeze_model(gan_model,recursive=True) # Does this have any role??
                        # my_model_save( gan_model           , os.path.join(save_dir,gan_model.name) )
                except:
                    print('Models Cannot be Saved')
                else:
                    print('Saved trained models')
                del datasets['LR'][name]['train'] , datasets['HR'][name]['train']
                iSUP, ia, ib = iSUP+SUP_delta, ia+a_delta, ib+b_delta
    del x_valid , y_valid, datasets['LR'][name]['valid'] , datasets['HR'][name]['valid']

######## TRAINING CODE SECTION ENDS ########


######## EVALUATING AND GENERATING MODEL OUTPUTS ON TEST BEGINS ########

def test(name):
    ""
    file_names[name] = {}
    file_names[name]['test'] = sorted(os.listdir(os.path.join(high_path,name,'test')))
    # Working each image at a time for full generation
    disk_batch = 1 # Taking disk_batch to be 1 as heterogenous images
    n_disk_batches = len(file_names[name]['test'][:test_images_limit])
    test_csv = open(os.path.join('.','..','experiments','{} {}.csv'.format(train_strategy,gan_model.name)),'w')
    print( 'index,img_name,initial_psnr,final_psnr,psnr_gain' , file=test_csv )
    test_csv.close()
    psnr_sum = {}
    init = datetime.now()
    new_generator_model = dict_to_model_parse( configs_dict[gen_choice] , (None,None,len(channel_indx)) )
    print('Time to Build New Model',datetime.now()-init,end='\n\n') # profiling
    init = datetime.now()
    if os.path.isdir(save_dir) and gen_choice in os.listdir(save_dir):
        my_model_load( new_generator_model ,  os.path.join(save_dir,gen_choice) )
    else:
        print('Model to Load for testing not found')
    print('Time to Load Weights',datetime.now()-init) # profiling
    if imwrite:
        gen_store = os.path.join(gen_path+train_strategy,name)
        if os.path.isdir(gen_store):
            shutil.rmtree(gen_store)
        if not os.path.isdir(gen_store):
            os.makedirs(gen_store)
    for i in range(n_disk_batches):
        gc.collect()
        init_time = datetime.now()
        feed_data(name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,False,'test',True) # erase all previous dataset
        print('Time to read image {} is {}'.format(i+1,datetime.now()-init_time))
        init = datetime.now()        
        x_test, y_test = get_data(name,'test',indx=['LR','HR'],org=False)
        print( 'Time to Scale Image',datetime.now()-init ) # profiling

        y_pred = new_generator_model.predict(x_test,verbose=1,batch_size=1)

        init = datetime.now()
        psnr_i =  get_IPSNR(np.array(y_test) , np.array(x_test) , pred_mode='LR')
        psnr_f =  get_IPSNR(np.array(y_test) , np.array(y_pred) , pred_mode='HR')
        psnr_g = psnr_f - psnr_i
        psnr_sum['initial'] = psnr_i if ('initial' not in psnr_sum) else (psnr_sum['initial']+psnr_i)
        psnr_sum['final'  ] = psnr_f if ('final'   not in psnr_sum) else (psnr_sum['final'  ]+psnr_f)
        psnr_sum['gain'   ] = psnr_g if ('gain'    not in psnr_sum) else (psnr_sum['gain'   ]+psnr_g)
        print('Time to Find and Store PSNRs',datetime.now()-init) # profiling
        print('Initial PSNR = {}, Final PSNR = {}, Gained PSNR = {}'.format(psnr_i,psnr_f,psnr_g))
        init = datetime.now()
        test_csv = open(os.path.join('.','..','experiments','{} {}.csv'.format(train_strategy,gan_model.name)),'a')
        print( '{},{},{},{},{}'.format(i+1,file_names[name]['test'][i],psnr_i,psnr_f,psnr_g) , file=test_csv )
        test_csv.close()
        print('Time to Update CSV file',datetime.now()-init) # profiling
        if imwrite and bit_depth==8:
            init = datetime.now()
            img_mat = backconvert(y_pred[0],typ='HR').astype('uint8')
            img = Image.fromarray( img_mat )
            img.save( os.path.join(gen_store,'{}.PNG'.format(file_names[name]['test'][i])) )
            # cv2.imwrite( os.path.join(gen_store,'{}.PNG'.format(file_names[name]['test'][i])) , cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB))
            print('Time to Write Image',datetime.now()-init) # profiling
        del x_test, y_test, y_pred, datasets['LR'][name]['test'], datasets['HR'][name]['test']
    # Finding average scored of learned model
    print('\nTest Statistics')
    for key in psnr_sum:
        print( 'Unweighted {:10s} PSNR = {}'.format(key,psnr_sum[key]/n_disk_batches) )

######## EVALUATING AND GENERATING MODEL OUTPUTS ON TEST ENDS ########


if __name__ == '__main__':
    # Building GAN model based on choices for both Training & Testing phase
    gan_model = my_gan( patch_size )
    datasets = {}
    if train_flag:
        # 
        train(data_name,'gan',(1,1))
    if test_flag:
        test(data_name)


######## UNUSED CODE SECTION BEGINS ########

"Below Code has Variable to Run in Google Colab"
'''
# Bool Type Arguments
train_flag, test_flag = True, True
prev_model = False
data_augmentation = False
imwrite = True
# Int Type Arguments
B = 4
patch_size, patches_limit = 96, None
outer_epochs, inner_epochs = 4, 5
disk_batch, memory_batch   = 10, 32
disk_batches_limit, valid_images_limit, test_images_limit = None, 5, None
# Float Type Arguments
lr, flr, decay, overlap = 0.0003, None, 0.0, 0.0
# String Type Arguments
data_name = 'DIV2K'
train_strategy = 'cnn'
gen_choice, dis_choice, con_choice = 'baseline_gen', 'baseline_dis', 'baseline_con'
'''