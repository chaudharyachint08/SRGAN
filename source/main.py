######## STANDARD & THIRD PARTY LIBRARIES IMPORTS BEGINS ########

# Importing Standard Python Libraries
ls = ['math','os','sys','argparse','datetime','shutil','itertools','struct']
for i in ls:
    exec('import {0}'.format(i))
    #exec('print("imported {0}")'.format(i))


os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
warnings.filterwarnings("ignore")


# Importing Standard Data Science & Deep Learning Libraries
ls = ['numpy','scipy','tensorflow','h5py','keras','sklearn','cv2','skimage']
for i in ls:
    exec('import {0}'.format(i))
    exec('print("Version of {0}",{0}.__version__)'.format(i))

import keras, tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# Keras models, sequential & much general functional API
from keras.models import Sequential, Model
# Core layers
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Reshape, Lambda, SpatialDropout2D
# Convolutional Layers
from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose
from keras.layers import UpSampling2D, ZeroPadding2D
# Pooling Layers
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
# Merge Layers
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
# Advanced Activation Layers
advanced_activations = ('ReLU','LeakyReLU','PReLU','ELU')
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
# Keras Model saving & Loading
from keras.engine.saving import save_model, load_model
# Keras Backend for additional functionalities
from keras import backend as K
K.set_image_data_format('channels_last')
# Used for image.resize(target_size,PIL.Image.BICUBIC)
import PIL
from PIL import Image
import os, gc, pickle, numpy as np
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from IPython.display import display
from numba import cuda

######## STANDARD & THIRD PARTY LIBRARIES IMPORTS ENDS ########


######## GLOBAL INITIAL VARIABLES BEGINS ########

parser = argparse.ArgumentParser()

# Bool Type Arguments
parser.add_argument("--run_main"           , type=eval , dest='run_main'           , default=False)
parser.add_argument("--train"              , type=eval , dest='train_flag'         , default=True)
parser.add_argument("--test"               , type=eval , dest='test_flag'          , default=True)
parser.add_argument("--prev_model"         , type=eval , dest='prev_model'         , default=False)
parser.add_argument("--change_optimizer"   , type=eval , dest='change_optimizer'   , default=True)
parser.add_argument("--data_augmentation"  , type=eval , dest='data_augmentation'  , default=False)
parser.add_argument("--imwrite"            , type=eval , dest='imwrite'            , default=False)
parser.add_argument("--read_as_gray"       , type=eval , dest='read_as_gray'       , default=False)
parser.add_argument("--gclip"              , type=eval , dest='gclip'              , default=False)
parser.add_argument("--epoch_lr_reduction" , type=eval , dest='epoch_lr_reduction' , default=False)

# Int Type Arguments
parser.add_argument("--scale"               , type=eval , dest='scale'               , default=4)
parser.add_argument("--patch_size"          , type=eval , dest='patch_size'          , default=96)
parser.add_argument("--patches_limit"       , type=eval , dest='patches_limit'       , default=None)
parser.add_argument("--patch_approach"      , type=eval , dest='patch_approach'      , default=1)
parser.add_argument("--min_LR"              , type=eval , dest='min_LR'              , default=0)
parser.add_argument("--max_LR"              , type=eval , dest='max_LR'              , default=1)
parser.add_argument("--min_HR"              , type=eval , dest='min_HR'              , default=-1)
parser.add_argument("--max_HR"              , type=eval , dest='max_HR'              , default=1)
parser.add_argument("--bit_depth"           , type=eval , dest='bit_depth'           , default=8)
parser.add_argument("--outer_epochs"        , type=eval , dest='outer_epochs'        , default=3)
parser.add_argument("--inner_epochs"        , type=eval , dest='inner_epochs'        , default=1)
parser.add_argument("--disk_batch"          , type=eval , dest='disk_batch'          , default=20)
parser.add_argument("--memory_batch"        , type=eval , dest='memory_batch'        , default=32)
parser.add_argument("--disk_batches_limit"  , type=eval , dest='disk_batches_limit'  , default=None)
parser.add_argument("--valid_images_limit"  , type=eval , dest='valid_images_limit'  , default=None)
parser.add_argument("--epoch_lr_red_epochs" , type=eval , dest='epoch_lr_red_epochs' , default=None)
parser.add_argument("--test_images_limit"   , type=eval , dest='test_images_limit'   , default=None)
parser.add_argument("--seed"                , type=eval , dest='np_seed'             , default=None)
parser.add_argument("--SUP"                 , type=eval , dest='SUP'                 , default=5)
parser.add_argument("--fSUP"                , type=eval , dest='fSUP'                , default=None)
parser.add_argument("--alpha"               , type=eval , dest='alpha'               , default=1)     # MS-SSIM specific
parser.add_argument("--beta"                , type=eval , dest='beta'                , default=1)     # MS-SSIM specific
parser.add_argument("--B"                   , type=eval , dest='B'                   , default=16)    # B blocks in Generator
parser.add_argument("--U"                   , type=eval , dest='U'                   , default=3)     # Recursive block's Units
# Float Type Arguments
parser.add_argument("--lr"                  , type=eval , dest='lr'                  , default=0.001) # Initial learning rate
parser.add_argument("--flr"                 , type=eval , dest='flr'                 , default=None)  # Final   learning rate
parser.add_argument("--decay"               , type=eval , dest='decay'               , default=0.0)   # Exponential Decay
parser.add_argument("--momentum"            , type=eval , dest='momentum'            , default=0.9)   # Momentum in case of SGD
parser.add_argument("--epoch_lr_red_factor" , type=eval , dest='epoch_lr_red_factor' , default=0.9)   # Recuction in LR on certain steps
parser.add_argument("--residual_scale"      , type=eval , dest='residual_scale'      , default=1)   # Recuction in LR on certain steps
parser.add_argument("--overlap"             , type=eval , dest='overlap'             , default=0.0)   # Explicit fraction of Overlap
parser.add_argument("--a"                   , type=eval , dest='a'                   , default=0.5)   # convexity b/w MS-SSIM & GL1
parser.add_argument("--fa"                  , type=eval , dest='fa'                  , default=None)  # convexity b/w MS-SSIM & GL1
parser.add_argument("--k1"                  , type=eval , dest='k1'                  , default=None)
parser.add_argument("--k2"                  , type=eval , dest='k2'                  , default=None)
parser.add_argument("--C1"                  , type=eval , dest='C1'                  , default=1)     # SSIM specific
parser.add_argument("--C2"                  , type=eval , dest='C2'                  , default=1)     # SSIM specific
parser.add_argument("--gnclip"              , type=eval , dest='gnclip'              , default=None)
parser.add_argument("--gvclip"              , type=eval , dest='gvclip'              , default=None)
parser.add_argument("--b"                   , type=eval , dest='b'                   , default=0.5)  # convexity b/w weighted & last
parser.add_argument("--fb"                  , type=eval , dest='fb'                  , default=None) # convexity b/w weighted & last
# String Type Arguments
parser.add_argument("--gen_choice"             , type=str  , dest='gen_choice'             , default='baseline_gen')
parser.add_argument("--dis_choice"             , type=str  , dest='dis_choice'             , default='baseline_dis')
parser.add_argument("--con_choice"             , type=str  , dest='con_choice'             , default='baseline_con')
parser.add_argument("--gen_loss"               , type=str  , dest='gen_loss'               , default=None)
parser.add_argument("--dis_loss"               , type=str  , dest='dis_loss'               , default=None)
parser.add_argument("--adv_loss"               , type=str  , dest='adv_loss'               , default=None)
parser.add_argument("--attention"              , type=str  , dest='attention'              , default='sigmoid')
parser.add_argument("--test_phase"             , type=str  , dest='test_phase'             , default='test')
parser.add_argument("--precision"              , type=str  , dest='float_precision'        , default='float32')
parser.add_argument("--optimizer1"             , type=str  , dest='optimizer1'             , default='Adam')
parser.add_argument("--optimizer2"             , type=str  , dest='optimizer2'             , default='Adam')
parser.add_argument("--name"                   , type=str  , dest='data_name'              , default='sample')
parser.add_argument("--train_strategy"         , type=str  , dest='train_strategy'         , default='cnn') # other is 'gan'
parser.add_argument("--high_path"              , type=str  , dest='high_path'              , default=os.path.join('.','..','data','HR'))
parser.add_argument("--low_path"               , type=str  , dest='low_path'               , default=os.path.join('.','..','data','LR'))
parser.add_argument("--gen_path"               , type=str  , dest='gen_path'               , default=os.path.join('.','..','data','SR'))
parser.add_argument("--save_dir"               , type=str  , dest='save_dir'               , default=os.path.join('.','..','experiments','saved_models'))
parser.add_argument("--plots"                  , type=str  , dest='plots'                  , default=os.path.join('.','..','experiments','training_plots'))
parser.add_argument("--input_interpolation"    , type=str  , dest='input_interpolation'    , default=None)
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
        if os.path.isdir(os.path.join(high_path,name,ph)):
            high_store = os.path.join(high_path,name,ph)
            low_store  = os.path.join(low_path, name,ph)
            
            gen_images = True
            if not os.path.isdir(low_store):
                os.makedirs(low_store)
                gen_images = True
            else: #Check scale at file
                with open(os.path.join(os.path.join(low_path, name),'scale.txt'),'r') as f:
                    if eval(f.read().strip())==scale:
                        gen_images = False
                    else:
                        gen_images = True
            if gen_images:
                with open(os.path.join(os.path.join(low_path, name),'scale.txt'),'w') as f:
                    f.write(str(scale))
                for img_name in os.listdir(high_store):
                    image = load_img(os.path.join(high_store,img_name),color_mode='rgb')
                    image = image.resize( ((image.width//scale),(image.height//scale)) ,
                        resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                    image.save(os.path.join(low_store,img_name))

def on_fly_crop(mat,patch_size=patch_size,typ='HR',overlap=overlap):
    "Crop images into patches, with explicit overlap, and limit of random patches if exist"
    ls = []
    # if input_interpolation:
    #     patch_size *= scale
    nparts1, nparts2 = int(np.ceil(mat.shape[0]/patch_size)) , int(np.ceil(mat.shape[1]/patch_size))
    step1, step2     = (mat.shape[0]-patch_size)/(nparts1-1) , (mat.shape[1]-patch_size)/(nparts2-1)
    if typ=='HR':
        step1, step2 = step1/scale, step2/scale
    step1, step2     = int(np.floor(step1*(1-overlap))) , int(np.floor(step2*(1-overlap)))
    for i in range(nparts1):
        i = round(i*step1)*(scale if typ=='HR' else 1)
        for j in range(nparts2):
            j = round(j*step2)*(scale if typ=='HR' else 1)
            ls.append( mat[i:i+patch_size,j:j+patch_size] )
    return np.array(ls)
    
def feed_data(name,low_path,high_path,lw_ix,up_ix,patching=True,phase='train',erase=False,patch_approach=patch_approach):
    "feed the required fraction of dataset into memory for consumption by model"
    global datasets
    if erase:
        datasets = {}
    check_and_gen(name,low_path,high_path)
    for x in ('LR','HR'):
        if x not in datasets:
            datasets[x] = {}
        if name not in datasets[x]:
            datasets[x][name] = {}
        # Entries for all phases, even if absent
        for ph in ('train','valid','test'):
            if (ph not in datasets[x][name]):
                datasets[x][name][ph] = []
        # Entry for present phase has to be LIST type, as to be append/extend
        datasets[x][name][phase] = []
    # PIL stores image in WIDTH, HEIGHT shape format, Numpy as HEIGHT, WIDTH
    clr = 'grayscale' if read_as_gray else 'rgb'
    for img_name in file_names[name][phase][lw_ix:up_ix]:
        image = load_img(os.path.join(high_path,name,phase,img_name),color_mode=clr)
        patch_ls, random_ix_ls = None, None
        for x,scl in zip(('HR','LR'),(scale,1)):
            if (not patching) or (patch_approach==1):
                image = image.resize( ((image.width//scale)*scl, (image.height//scale)*scl) ,
                    resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                if (patch_approach==1) and (input_interpolation is not None):
                    image = image.resize( (image.width*scale, image.height*scale) ,
                        resample = eval('PIL.Image.{}'.format(input_interpolation)) )
            if not ( patching and x=='LR' and patch_approach==2 ):
                img_mat = img_to_array( image , dtype='uint{}'.format(bit_depth))
            if not patching:
                datasets[x][name][phase].append(img_mat)
            else: # If patches are to be used, for training & validation data
                if x=='HR':
                    patch_ls = on_fly_crop( img_mat,patch_size,x )
                    if patches_limit is not None:
                        random_ix_ls = np.arange(len(patch_ls))
                        np.random.shuffle(random_ix_ls)
                        random_ix_ls = random_ix_ls[:patches_limit]
                        patch_ls = patch_ls[random_ix_ls]
                    datasets[x][name][phase].extend(patch_ls)
                else: # There are 2 approaches for generating LR patches
                    if patch_approach == 1 :
                        patch_ls = on_fly_crop( img_mat,patch_size//(scale if (input_interpolation is None) else 1),x )
                        if patches_limit is not None:
                            patch_ls = patch_ls[random_ix_ls]
                        datasets[x][name][phase].extend(patch_ls)
                    elif patch_approach == 2 :
                        for mat in patch_ls:
                            patch_image = Image.fromarray(mat)
                            patch_image = image.resize( ((patch_image.width//scale),(patch_image.height//scale)) ,
                                resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                            if input_interpolation is not None:
                                patch_image = image.resize( ((patch_image.width*scale),(patch_image.height*scale)) ,
                                    resample = eval('PIL.Image.{}'.format(input_interpolation)) )
                            mat = img_to_array( patch_image , dtype='uint{}'.format(bit_depth))
                            datasets[x][name][phase].append(mat)
    for x in ('LR','HR'):
        datasets[x][name][phase] = np.array(datasets[x][name][phase])

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

def show_ix(mat,ix,preproc=True,typ='HR',ipython=True):
    if preproc:
        img = Image.fromarray(  backconvert(mat[ix],typ).astype('uint8')  )
    else:
        img = Image.fromarray(  mat[ix].astype('uint8')  )
    if ipython:
        display(img)
    else:
        img.show()

def show_patches(name,ix=0,images_limit=5,patches_limit = 20,patch_approach=1):
    "Function to observe patches made from with of approach in datasets"
    global datasets, file_names

    train_strategy = 'cnn'

    datasets = {}
    datasets[name] = {}
    file_names[name] = {}

    file_names[name]['test'] = sorted(os.listdir(os.path.join(high_path,name,'test')))
    np.random.shuffle(file_names[name]['test'])
    feed_data(name,low_path,high_path,0,images_limit,True,'test',False,patch_approach)

    x_test  = get_data(name,'test',indx=['LR','HR'],org=False)
    x_test  = {    'lr_input': x_test[0] ,    'hr_input': x_test[1]    }
    init_PSNR = get_PSNR(x_test['hr_input'],x_test['lr_input'],pred_mode='LR')
    x_test, y_test = x_test['lr_input'], x_test['hr_input']

    ls = np.arange(len(x_test))
    np.random.shuffle(ls)
    ls = ls[:patches_limit]

    for ix in ls:
        show_ix(x_test,ix,typ='LR')
        show_ix(y_test,ix,typ='HR')
        # show_ix(datasets['LR'][name]['valid'],ix,typ='HR',preproc=False)
        # show_ix(datasets['HR'][name]['valid'],ix,typ='HR',preproc=False)
        print()

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

def get_PSNR(true_tensor,pred_tensor,pred_mode='HR'):
    "return average PSNR between two tensors containing Images"
    avg_val = [0,0]
    if (pred_mode=='HR') or (pred_mode=='LR' and bit_depth==8):
        for i in range(len(true_tensor)):
            true_img = backconvert(true_tensor[i] , typ='HR' )
            pred_img = backconvert(pred_tensor[i] , typ=pred_mode )
            # PIL supports only 8 bit wide images for now, makes sense so restrict initial PSNR
            if pred_mode=='LR':
                pred_img = pred_img.astype('uint8')
                pred_img = Image.fromarray(pred_img)
                if input_interpolation is None:
                    pred_img = pred_img.resize( ((pred_img.width*scale),(pred_img.height*scale)) ,
                        resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                pred_img = img_to_array(pred_img)
    #         val = npPSNR(true_img,pred_img)
    #         if val!=float('inf'):
    #             avg_val[0] += val
    #             avg_val[1] += 1
    # return (avg_val[0]/(avg_val[1]) if avg_val[1] else avg_val[0])

            val = np.mean( np.square(true_img-pred_img) )
            avg_val[0] += val
            avg_val[1] += 1
    return (10.0 * np.log((((1<<bit_depth)-1) ** 2) / (avg_val[0]/avg_val[1]))) / np.log(10.0)


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
for loss in ('MIX_LOSS','ZERO_LOSS','DIS_LOSS','ADV_LOSS',):
    keras_update_dict[loss] = eval(loss)
# for layer in ('WeightedSumLayer','PixelShuffle','DePixelShuffle'):
#     keras_update_dict[layer] = eval(layer)
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
            elif lyr_typ in ('convo','usmpl','zpdng'):
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
                nxt_lyr  = eval('WeightedSumLayer')
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
    if input_interpolation is None:
        generator_model     = dict_to_model_parse( configs_dict[gen_choice],(shapes[0]//scale , shapes[1]//scale , shapes[2]) )
    else:
        generator_model     = dict_to_model_parse( configs_dict[gen_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    discriminator_model = dict_to_model_parse( configs_dict[dis_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    content_model       = dict_to_model_parse( configs_dict[con_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    generator_model.name = 'generator'
    if input_interpolation is None:
        X_lr = Input((shapes[0]//scale , shapes[1]//scale , shapes[2]),name='lr_input')
    else:
        X_lr = Input((shapes[0]      , shapes[1]        , shapes[2]),name='lr_input')
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

def compile_model(model,mode,opt):
    if mode in ('cnn','gen'):
        train_model , non_train_model = generator_model, discriminator_model
        loss = { 'generator':('MAE' if (gen_loss is not None) else gen_loss) , 'discriminator':('ADV_LOSS' if (adv_loss is not None) else adv_loss) , 'content':'ZERO_LOSS' }
        if mode=='cnn': # Although this one is not used anymore
            loss_weights = { 'generator':1 , 'discriminator':0 , 'content':0 }
        elif mode=='gen':
            loss_weights = { 'generator':1e-2 , 'discriminator':5e-3 , 'content':(1/12.75)**2 }
    elif mode=='dis':
        non_train_model , train_model = generator_model, discriminator_model
        loss = { 'generator':('MAE' if (gen_loss is not None) else gen_loss) , 'discriminator':('DIS_LOSS' if (dis_loss is not None) else dis_loss) , 'content':'ZERO_LOSS' }
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
            with open(os.path.join(plot_dir,'{}.txt'.format(key)),'w') as wf:
                wf.write( '='.join( map( str , ( key        , history[key]        )) )+'\n' )
                wf.write( '='.join( map( str , ( 'val_'+key , history['val_'+key] )) )      )
            _ = plt.plot( history[key]        , linewidth=1 , label=key )
            _ = plt.plot( history['val_'+key] , linewidth=1 , label='val_'+key )
            plt.xticks( np.round(np.linspace(0,len(history[key])-1,5),0) , np.round(np.linspace(1,len(history[key]),5),0) )
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3g}')) # 2 decimal places
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
    global x_valid, y_valid, x_train, y_train
    global generator_model, discriminator_model, content_model, gan_model
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
    feed_data(name,low_path,high_path,None,valid_images_limit,patching=True,phase='valid',erase=False,patch_approach=patch_approach)
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
    val_init_PSNR = get_PSNR(x_valid['hr_input'],x_valid['lr_input'],pred_mode='LR')
    if train_strategy=='cnn':
        x_valid, y_valid = x_valid['lr_input'], x_valid['hr_input']
    # Selecting optimizers for training
    if optimizer1 == 'Adam':
        opt1 = keras.optimizers.Adam( lr=lr, decay=decay )
    elif optimizer1 == 'SGD':
        opt1 = keras.optimizers.SGD( lr=lr, decay=decay, momentum=momentum )
    else:   
        raise Exception('Unexpected Optimizer1 than two classics')
    if optimizer2 == 'Adam':
        opt2 = keras.optimizers.Adam( lr=lr, decay=decay )
    elif optimizer2 == 'SGD':
        opt2 = keras.optimizers.SGD( lr=lr, decay=decay, momentum=momentum )
    else:   
        raise Exception('Unexpected Optimizer2 than two classics')
    # collecting history from all opechs, and oter_epochs training loop
    all_history = None
    for epc in range(outer_epochs):
        print('\nOuter Epoch {}'.format(epc+1))
        mode_ix = 0
        while mode_ix < len(modes):
            if train_strategy=='cnn' and (not epc):
                if prev_model and os.path.isfile(os.path.join(save_dir,gen_choice)):
                    if change_optimizer:
                        generator_model.compile( optimizer=opt1, loss='MSE', metrics=['PSNR'] )
                        generator_model.load_weights(     os.path.join(save_dir,gen_choice) )
                    else:
                        generator_model = load_model( os.path.join(save_dir,gen_choice),
                            custom_objects=keras_update_dict, compile=True )
                else:
                    generator_model.compile( optimizer=opt1, loss='MSE', metrics=['PSNR'] )
                model = generator_model
            elif train_strategy=='gan':
                print( 'Executing GAN in {} mode'.format('GENERATOR' if modes[mode_ix]=='gen' else 'DISCRIMINATOR') )
                if modes[mode_ix] != modes[(mode_ix-1)%len(modes)]:
#                    K.clear_session()
#                    if tf.test.is_gpu_available():
#                       cuda.select_device(0) ; cuda.close()
#                     for obj_name in ('generator_model', 'discriminator_model', 'content_model', 'gan_model', 'model'):
#                         try:
#                             exec('del {}'.format(x),locals(),globals())
#                         except:
#                             pass
#                     gc.collect()

                    if os.path.isfile(os.path.join(save_dir,'-'.join((gan_model.name,modes[mode_ix])))):
                        if (not epc) and change_optimizer:
                            compile_model( gan_model, modes[mode_ix], (opt1 if modes[mode_ix]=='gen' else opt2) )
                        else:
                            gan_model = load_model( os.path.join(save_dir,'-'.join((gan_model.name,modes[mode_ix]))),
                                custom_objects=keras_update_dict, compile=True )
                    else:
                        compile_model( gan_model, modes[mode_ix], (opt1 if modes[mode_ix]=='gen' else opt2) )
                    if os.path.isfile(os.path.join(save_dir,gen_choice)):
                        generator_model.load_weights(     os.path.join(save_dir,gen_choice) )
                    if os.path.isfile(os.path.join(save_dir,dis_choice)):
                        discriminator_model.load_weights( os.path.join(save_dir,dis_choice) )
                    model = gan_model
                    
            np.random.shuffle(file_names[name]['train'])
            iSUP = int(np.round(iSUP)) if int(np.round(iSUP))%2 else int(np.round(iSUP))+1
            
            for i in range(n_disk_batches):
                gc.collect()
                K.set_value(model.optimizer.lr    , lr+(i+epc*n_disk_batches)*lr_delta)
                K.set_value(model.optimizer.decay , decay)
                if epoch_lr_reduction and (not (i+epc*n_disk_batches)) and (not ((i+epc*n_disk_batches)%epoch_lr_red_epochs)):
                    K.set_value(model.optimizer.lr    , K.eval(model.optimizer.lr)*epoch_lr_red_factor)
                if gclip:
                    if gnclip is not None:
                        model.optimizer.__dict__['clipnorm']  = gnclip / K.eval(model.optimizer.lr)
                    if gvclip is not None:
                        model.optimizer.__dict__['clipvalue'] = gvclip / K.eval(model.optimizer.lr)
                
                print( 'Reading Disk Batch {}'.format(i+1) )
                init_time = datetime.now()
                feed_data(name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,patching=True,phase='train',erase=False,patch_approach=patch_approach)
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
                train_init_PSNR = get_PSNR(x_train['hr_input'],x_train['lr_input'],pred_mode='LR')
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
                del x_train, y_train, datasets['LR'][name]['train'] , datasets['HR'][name]['train']
                all_history = history if (all_history is None) else { i:(all_history[i]+history[i])for i in history }
                plot_history(all_history) # Plotting progress after each disk batch
                #Saving Trained model, after all QP values of currect disk_batch were processed
                print( '\nSaving Trained model > ',end='' )
                # Save model and weights
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                try:
                    if train_strategy=='cnn':
                        save_model(generator_model, os.path.join(save_dir,gen_choice), overwrite=True, include_optimizer=True)
                    elif train_strategy=='gan':
                        save_model(generator_model,     os.path.join(save_dir,gen_choice), overwrite=True, include_optimizer=False)
                        save_model(discriminator_model, os.path.join(save_dir,dis_choice), overwrite=True, include_optimizer=False)
                        if modes[mode_ix] != modes[(mode_ix+1)%len(modes)]:
                            save_model(gan_model, os.path.join(save_dir,'-'.join((gan_model.name,modes[mode_ix]))), overwrite=True, include_optimizer=True)
                except:
                    print('Models & Optimizers Cannot be Saved')
                else:
                    print('Saved trained models')
                iSUP, ia, ib = iSUP+SUP_delta, ia+a_delta, ib+b_delta
                if train_strategy=='gan':
                    if modes[mode_ix] != modes[(mode_ix+1)%len(modes)]:
                        if   modes[mode_ix]=='dis':
                            gen_loss = all_history['generator_loss'    ][-1]
                            dis_loss = all_history['discriminator_loss'][-1]
                            con_loss = all_history['content_loss'      ][-1]
                            if dis_loss > (-1*np.log(0+60/100)):
                                print('Resetting to Discriminator',dis_loss,(-1*np.log(0+60/100)))
                                mode_ix -= 1
                        elif modes[mode_ix]=='gen':
                            gen_loss = all_history['generator_loss'    ][-1]
                            dis_loss = all_history['discriminator_loss'][-1]
                            con_loss = all_history['content_loss'      ][-1]
#                            if dis_loss > (-1*np.log(1-60/100)):
#                                 print('Resetting to Generator',dis_loss,(-1*np.log(1-60/100)))
#                                 mode_ix -= 1
            mode_ix += 1
    del x_valid , y_valid, datasets['LR'][name]['valid'] , datasets['HR'][name]['valid']

######## TRAINING CODE SECTION ENDS ########


######## EVALUATING AND GENERATING MODEL OUTPUTS ON TEST BEGINS ########

def test(name,phase=test_phase):
    ""
    # K.clear_session()
    # if tf.test.is_gpu_available():
    #     cuda.select_device(0) ; cuda.close()
    file_names[name] = {}
    file_names[name][phase] = sorted(os.listdir(os.path.join(high_path,name,phase)))
    # Working each image at a time for full generation
    disk_batch = 1 # Taking disk_batch to be 1 as heterogenous images
    n_disk_batches = len(file_names[name][phase][:test_images_limit])
    test_csv = open(os.path.join('.','..','experiments','{} {} {} {}.csv'.format(name,phase,train_strategy,gan_model.name)),'w')
    print( 'index,img_name,initial_psnr,final_psnr,psnr_gain' , file=test_csv )
    test_csv.close()
    psnr_sum = {}
    init = datetime.now()
    new_generator_model = dict_to_model_parse( configs_dict[gen_choice] , (None,None,len(channel_indx)) )
    print('Time to Build New Model',datetime.now()-init,end='\n\n') # profiling
    init = datetime.now()
    if os.path.isdir(save_dir) and '{}'.format(gen_choice) in os.listdir(save_dir):
        new_generator_model.load_weights(os.path.join(save_dir,gen_choice))
    else:
        print('Model to Load for testing not found')
    print('Time to Load Weights',datetime.now()-init) # profiling
    if imwrite:
        gen_store = os.path.join(gen_path+train_strategy,gan_model.name,name,phase)
        if os.path.isdir(gen_store):
            shutil.rmtree(gen_store)
        if not os.path.isdir(gen_store):
            os.makedirs(gen_store)
    print( '{} Images to Test on'.format((n_disk_batches)) )
    for i in range(n_disk_batches):
        gc.collect()
        init_time = datetime.now()
        feed_data(name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,patching=False,phase=phase,erase=True,patch_approach=patch_approach)
        print('Time to read image {}\n{} is {}'.format(file_names[name][phase][i],i+1,datetime.now()-init_time))
        init = datetime.now()        
        x_test, y_test = get_data(name,phase,indx=['LR','HR'],org=False)
        print( 'Time to Scale Image',datetime.now()-init ) # profiling

        y_pred = new_generator_model.predict(x_test,verbose=1,batch_size=1)

        init = datetime.now()
        psnr_i =  get_PSNR(np.array(y_test) , np.array(x_test) , pred_mode='LR')
        psnr_f =  get_PSNR(np.array(y_test) , np.array(y_pred) , pred_mode='HR')
        psnr_g = psnr_f - psnr_i
        psnr_sum['initial'] = psnr_i if ('initial' not in psnr_sum) else (psnr_sum['initial']+psnr_i)
        psnr_sum['final'  ] = psnr_f if ('final'   not in psnr_sum) else (psnr_sum['final'  ]+psnr_f)
        psnr_sum['gain'   ] = psnr_g if ('gain'    not in psnr_sum) else (psnr_sum['gain'   ]+psnr_g)
        print('Time to Find and Store PSNRs',datetime.now()-init) # profiling
        print('Initial PSNR = {}, Final PSNR = {}, Gained PSNR = {}'.format(psnr_i,psnr_f,psnr_g))
        init = datetime.now()
        test_csv = open(os.path.join('.','..','experiments','{} {} {} {}.csv'.format(name,phase,train_strategy,gan_model.name)),'a')
        print( '{},{},{},{},{}'.format(i+1,file_names[name][phase][i],psnr_i,psnr_f,psnr_g) , file=test_csv )
        test_csv.close()
        print('Time to Update CSV file',datetime.now()-init) # profiling
        if imwrite and bit_depth==8:
            init = datetime.now()
            img_mat = backconvert(y_pred[0],typ='HR').astype('uint8')
            img = Image.fromarray( img_mat )
            img.save( os.path.join(gen_store,'{}'.format(file_names[name][phase][i])) )
            print('Time to Write Image',datetime.now()-init) # profiling
        del x_test, y_test, y_pred, datasets['LR'][name][phase], datasets['HR'][name][phase]
    # Finding average scored of learned model
    print('\nTest Statistics')
    for key in psnr_sum:
        print( 'Unweighted {:10s} PSNR = {}'.format(key,psnr_sum[key]/n_disk_batches) )

######## EVALUATING AND GENERATING MODEL OUTPUTS ON TEST ENDS ########







if (__name__ == '__main__') and run_main:
    # Bool Type Arguments
    imwrite = True
    train_flag, test_flag = True, False
    prev_model, change_optimizer, data_augmentation = False, False, False
    # Int Type Arguments
    patches_limit = None
    patch_approach = 1
    min_LR, max_LR = 0,1
    min_HR, max_HR = -1,1
    outer_epochs, inner_epochs = 10, 3
    disk_batch, memory_batch   = 10, 8
    disk_batches_limit, valid_images_limit, test_images_limit = 5, 15, 10
    # Float Type Arguments
    overlap = 0.0
    lr, flr, decay, momentum = 100e-5, 10e-5, 0.0, 0.8
    # String Type Arguments
    optimizer1 = 'Adam'
    train_strategy = 'cnn'
    data_name = 'DIV2K'
    # Building GAN model based on choices for both Training & Testing phase
    gan_model = my_gan( patch_size )
    datasets = {}
    if train_flag:
        train(data_name,train_strategy,(1,1))
    if test_flag:
        test(data_name)

else:
    gan_model = my_gan( patch_size )
    datasets = {}

    if train_flag:
        train(data_name,train_strategy,(1,1))
    if test_flag:
        test(data_name)


















######## UNUSED CODE SECTION BEGINS ########


'''
"Below Code has Variable to Run in Google Colab"
B = 16
initiate(globals())
patch_size = 128
gen_choice, dis_choice, con_choice = 'baseline_gen', 'baseline_dis', 'baseline_con'
gan_model = my_gan( patch_size )
'''


'''
"Code for De-bugging purpose of GAN generation for varying scale"
print('Generator     I/O Shapes')
print( generator_model.layers[ 0].input_shape  )
print( generator_model.layers[-1].output_shape )
print('Discriminator I/O Shapes')
print( discriminator_model.layers[ 0].input_shape  )
print( discriminator_model.layers[-1].output_shape )
print('Content       I/O Shapes')
print( content_model.layers[ 0].input_shape  )
print( content_model.layers[-1].output_shape )
'''


'''
"SRResNet TRAINING FOR GOOGLE COLAB"
# Bool Type Arguments
imwrite = True
train_flag, test_flag = True, True
prev_model, change_optimizer, data_augmentation = False, False, False
# Int Type Arguments
patches_limit = None
patch_approach = 1
min_LR, max_LR = 0,1
min_HR, max_HR = -1,1
outer_epochs, inner_epochs = 3, 5
disk_batch, memory_batch   = 5, 32
disk_batches_limit, valid_images_limit, test_images_limit = 3, 15, 10
# Float Type Arguments
overlap = 0.0
lr, flr, decay, momentum = 100e-5, 10e-5, 0.0, 0.8
# String Type Arguments
optimizer1 = 'Adam'
train_strategy = 'cnn'
data_name = 'DIV2K'

datasets = {}
if train_flag:
    train(data_name,train_strategy,(1,1))
if test_flag:    
    test(data_name)
'''


'''
"SRGAN TRAINING FOR GOOGLE COLAB"
# Bool Type Arguments
imwrite = True
train_flag, test_flag = True, True
prev_model, change_optimizer, data_augmentation = False, False, False
# Int Type Arguments
patches_limit = None
patch_approach = 1
min_LR, max_LR = 0,1
min_HR, max_HR = -1,1
outer_epochs, inner_epochs = 2, 3
disk_batch, memory_batch   = 10, 32
disk_batches_limit, valid_images_limit, test_images_limit   = 5, 5, 10
# Float Type Arguments
overlap = 0.0
lr, flr, decay, momentum = 5e-5, None, 0.0, 0.8
# String Type Arguments
optimizer1 = 'Adam'
optimizer2 = 'Adam'
train_strategy = 'gan'
data_name = 'DIV2K'

datasets = {}
if train_flag:
    train(data_name,train_strategy,(1,1))
if test_flag:    
    test(data_name)
'''





# def model_save(model,filepath):
#     save_model(model, "{}".format(filepath), overwrite=True, include_optimizer=True)
#     with open("{}.json".format(name), "w") as json_file:
#         json_file.write( model.to_json() )
#     model.save_weights("{}.h5".format(name))

# def model_load(filepath,custom_objects,compile=True):
#     return load_model("{}".format(filepath), custom_objects=custom_objects, compile=compile)
#     model.load_weights('{}.h5'.format(filepath))

# def opt_save(obj,filepath):
#     with open(filepath,'wb') as f:
#         pickle.dump(obj,f)

# def opt_load(filepath):
#     with open(filepath,'rb') as f:
#         obj = f.load()
#     return obj
