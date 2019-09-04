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
# keras image usage functions
from keras.preprocessing.image import load_img, img_to_array

# Used for image.resize(target_size,PIL.Image.BICUBIC)
import PIL
from PIL import Image

# Keras Backend for additional functionalities
from keras import backend as K

K.set_image_data_format('channels_last')


import os, numpy as np
from skimage import io
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

######## STANDARD & THIRD PARTY LIBRARIES IMPORTS ENDS ########


######## GLOBAL INITIAL VARIABLES BEGINS ########

parser = argparse.ArgumentParser()

# Bool Type Arguments
parser.add_argument("--train",             type=eval, dest='train_flag',        default=True)
parser.add_argument("--test",              type=eval, dest='test_flag',         default=True)
parser.add_argument("--custom",            type=eval, dest='custom_model',      default=True)
parser.add_argument("--prev_model",        type=eval, dest='prev_model',        default=False)
parser.add_argument("--imwrite",           type=eval, dest='imwrite',           default=False)
parser.add_argument("--gclip",             type=eval, dest='gclip',             default=False)
parser.add_argument("--mix_diff_qps",      type=eval, dest='mix_diff_qps',      default=True)
parser.add_argument("--read_as_gray",      type=eval, dest='read_as_gray',      default=True)
parser.add_argument("--data_augmentation", type=eval, dest='data_augmentation', default=False)

# Int Type Arguments
parser.add_argument("--scale",              type=eval, dest='scale',               default=4)
parser.add_argument("--block_size",         type=eval, dest='block_size',          default=96)
parser.add_argument("--bit_depth",          type=eval, dest='bit_depth',           default=8)
parser.add_argument("--min_LR",             type=eval, dest='min_LR',              default=0)
parser.add_argument("--max_LR",             type=eval, dest='max_LR',              default=1)
parser.add_argument("--min_HR",             type=eval, dest='min_HR',              default=-1)
parser.add_argument("--max_HR",             type=eval, dest='max_HR',              default=1)

parser.add_argument("--outer_epochs",       type=eval, dest='outer_epochs',        default=5)
parser.add_argument("--inner_epochs",       type=eval, dest='inner_epochs',        default=2)
parser.add_argument("--disk_batches_limit", type=eval, dest='disk_batches_limit',  default=None)
parser.add_argument("--valid_images_limit", type=eval, dest='valid_images_limit',  default=None)
parser.add_argument("--test_images_limit",  type=eval, dest='test_images_limit',   default=None)
parser.add_argument("--disk_batch",         type=eval, dest='disk_batch',          default=20)
parser.add_argument("--memory_batch",       type=eval, dest='memory_batch',        default=32)

parser.add_argument("--seed",               type=eval, dest='np_seed',             default=None)
parser.add_argument("--SUP",                type=eval, dest='SUP',                 default=5)
parser.add_argument("--fSUP",               type=eval, dest='fSUP',                default=None)
parser.add_argument("--alpha",              type=eval, dest='alpha',               default=1) # MS-SSIM specific
parser.add_argument("--beta",               type=eval, dest='beta',                default=1) # MS-SSIM specific
parser.add_argument("--U",                  type=eval, dest='U',                   default=3) # Units in a recursive block
parser.add_argument("--B",                  type=eval, dest='B',                   default=1) # Block in drnn network

# Float Type Arguments
parser.add_argument("--lr",       type=eval, dest='lr',       default=0.001) # Initial learning rate
parser.add_argument("--flr",      type=eval, dest='flr',      default=None) # Final Target learning rate, linear schedule
parser.add_argument("--decay",    type=eval, dest='decay',    default=0.0) # Exponential Decay paramter
parser.add_argument("--momentum", type=eval, dest='momentum', default=0.9) # Exponential Decay paramter
parser.add_argument("--overlap",  type=eval, dest='overlap',  default=0.0) #Fractional Overlap between two image blocks
parser.add_argument("--a",        type=eval, dest='a',        default=0.5) # 'a' as convexity between MS-SSIM & GL1
parser.add_argument("--fa",       type=eval, dest='fa',       default=None) # 'a' as convexity between MS-SSIM & GL1
parser.add_argument("--k1",       type=eval, dest='k1',       default=None)
parser.add_argument("--k2",       type=eval, dest='k2',       default=None)
parser.add_argument("--C1",       type=eval, dest='C1',       default=1)
parser.add_argument("--C2",       type=eval, dest='C2',       default=1)
parser.add_argument("--gnclip",   type=eval, dest='gnclip',   default=None)
parser.add_argument("--gvclip",   type=eval, dest='gvclip',   default=None)

# CodeBase not ready for Multioutput models yet
parser.add_argument("--b",        type=eval, dest='b',        default=0.5) # NU 'b' as recursion weighted average & last output
parser.add_argument("--fb",       type=eval, dest='fb',       default=None) # NU 'b' as recursion weighted average & last outputd 

# String Type Arguments
parser.add_argument("--resize_interpolation",   type=str, dest='resize_interpolation',    default='BICUBIC')
parser.add_argument("--upsample_interpolation", type=str, dest='upsample_interpolation',  default='bicubic')

parser.add_argument("--gen_choice", type=str, dest='gen_choice',    default='baseline_gen')
parser.add_argument("--dis_choice", type=str, dest='dis_choice',    default='baseline_dis')
parser.add_argument("--con_choice", type=str, dest='con_choice',    default='baseline_con')

# parser.add_argument("--loss",         type=str, dest='loss',            default='MSE')
parser.add_argument("--attention",    type=str, dest='attention',       default='sigmoid')
parser.add_argument("--precision",    type=str, dest='float_precision', default='float32')
parser.add_argument("--optimizer",    type=str, dest='optimizer',       default='Adam')
parser.add_argument("--name",         type=str, dest='data_name',       default='sample')
parser.add_argument("--save_dir",     type=str, dest='save_dir',        default='saved_models')
parser.add_argument("--save_dir",     type=str, dest='train_strategy',  default='cnn') # Default is for CNN based SR other is 'gan'
parser.add_argument("--high_path",    type=str, dest='high_path',       default=os.path.join('.','..','data','HR'))
parser.add_argument("--low_path",     type=str, dest='low_path',        default=os.path.join('.','..','data','LR'))
parser.add_argument("--gen_path",     type=str, dest='gen_path',        default=os.path.join('.','..','data','SR'))

# Tuple Type Arguments
parser.add_argument("--channel_indx",  type=eval, dest='channel_indx',  default=(0,1,2))

args = parser.parse_args()
globals().update(args.__dict__)

######## GLOBAL INITIAL VARIABLES ENDS ########


######## CUSTOM IMPORTS AFTER COMMAND LINE ARGUMENTS PARSING BEGINS ########

import models_collection
models_collection.initiate(globals())
from models_collection import configs_dict

######## CUSTOM IMPORTS AFTER COMMAND LINE ARGUMENTS PARSING ENDS ########


######## PROGRAM INITILIZATION BEGINS ########

np.random.seed(np_seed)
C1, C2 = k1**2 if (k1 is not None) else C1 , k2**2 if (k2 is not None) else C2

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
                    clr = 'grayscale' if read_as_gray else 'rgb'
                    image = load_img(os.path.join(high_store,img_name),color_mode=clr)
                    image = image.resize( ((image.width//scale),(image.height//scale)) ,
                        resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                    image.save(os.path.join(low_store,img_name))

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
                for img_name in file_names[name][phase][lw_indx:up_indx]:
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

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def DIS_LOSS(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def GEN_LOSS(y_true, y_pred):
    y_pred = 1-y_pred
    return K.categorical_crossentropy(y_true, y_pred)

custom_losses = ('MIX_LOSS','ZERO_LOSS','DIS_LOSS','GEN_LOSS',)

######## CUSTOM LOSS FUNCTIONS DEFINITIONS ENDS ########


######## METRIC DEFINITIONS BEGINS ########

def PSNR(y_true, y_pred):
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
                pred_img = Image.fromarray(pred_img)
                pred_img = pred_img.resize( ((pred_image.width*scale),(pred_image.height*scale)) ,
                    resample = eval('PIL.Image.{}'.format(resize_interpolation)) )
                pred_img = img_to_array(pred_img)
            avg_PSNR[0] += npPSNR(true_img,pred_img)
            avg_PSNR[1] += 1
    return (avg_PSNR[0]/(avg_PSNR[1]) if avg_PSNR[1] else avg_PSNR[0])

######## METRIC DEFINITIONS ENDS ########


######## KERAS FUNCTIONS A UPDATE BEGINS ########

keras_update_dict = {}
for loss in custom_losses:
    keras_update_dict[loss] = eval(loss)
keras_update_dict['PSNR'] = eval('PSNR')
get_custom_objects().update( keras_update_dict)

######## KERAS FUNCTIONS A UPDATE ENDS ########



######## MODELS DEFINITIONS BEGINS ########

def baseline(shape1,shape2): # 2 channels, as one from Y/U/V and second if Normalizaed QP map
    X_input1 = Input(shape1)
    X_input2 = Input(shape2)
    X_input = Concatenate()([X_input1, X_input2])
    X = Conv2D(64, (5,5), strides=(1, 1), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(X_input)
    conv = Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
    for i in range(6):
        X = conv(X)
    X = Conv2D(1, (3,3), strides=(1, 1), padding='same', activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Add()([X_input1, X])
    model = Model(inputs = [X_input1, X_input2], outputs = X, name='baseline2')
    return model

def dict_to_model_parse(config,*ip_shapes):
    # Recorrecting order issues, is so exist in input config
    config['l_conv_connection']  = [ tuple(sorted(i)) for i in config['l_conv_connection']  ] if 'l_conv_connection'  in config else []
    config['l_merge_connection'] = [ tuple(sorted(i)) for i in config['l_merge_connection'] ] if 'l_merge_connection' in config else []
    config['l_block_connection'] = [ tuple(sorted(i)) for i in config['l_block_connection'] ] if 'l_block_connection' in config else []
    # CodeBase not ready for Multioutput models yet
    config['outputs']            = [ tuple(sorted(i)) for i in config['outputs'] ]            if 'outputs'            in config else []

    #Inputs are primary tensors
    target, tensors = len(ip_shapes), [ Input(i) for i in ip_shapes ]
    while True:
        ls_conv  = [ i[-1] for i in config['l_conv_connection']  ] if 'l_conv_connection'  in config else []
        ls_merge = [ i[-1] for i in config['l_merge_connection'] ] if 'l_merge_connection' in config else []
        ls_block = [ i[-1] for i in config['l_block_connection'] ] if 'l_block_connection' in config else []
        if target in ls_merge:
            src_indx = ls_merge.index( target )
            lyr = eval( '{}()'.format(config['l_merge_type'][src_indx]) )
            ip = config['l_merge_connection'][src_indx][:-1]
            tensors.append( lyr([tensors[i] for i in ip]) )
        elif target in ls_conv:
            src_indx = ls_conv.index( target )
            ls_args = []
            for i in ('filters','kernel_size','strides','padding','dilation_rate','activation'):
                if (i!='activation') or (config['l_'+i][src_indx] not in advanced_activations):
                    ls_args.append( '{}={}'.format(i,repr(config['l_'+i][src_indx])) )
            args_str = ', '.join(ls_args)
            lyr = eval( '{}({})'.format(config['l_conv_type'][src_indx],args_str) )
            ip = config['l_conv_connection'][ ls_conv.index( target )][0]
            if config['l_activation'][src_indx] not in advanced_activations:        
                tensors.append( lyr(tensors[ip]) )
            else:
                tensors.append( eval(config['l_activation'][src_indx])()(lyr(tensors[ip])) )
        elif target in ls_block:
            src_indx = ls_block.index( target )
            tensors.append( configs_dict[config['l_block_fun'][src_indx]](*(tensors[i] for i in config['l_block_connection'][src_indx][:-1])) )
        else:
            break
        target += 1
    if config['outputs']: # NU : CodeBase not ready for Multioutput models yet
        model = Model(inputs = tensors[:len(ip_shapes)], outputs = [tensors[i] for i in config['outputs']], name=config['name'])
    else:
        model = Model(inputs = tensors[:len(ip_shapes)], outputs = tensors[-1], name=config['name'])

    return model


def my_gan(*shapes):
    global generator_model, discriminator_model, content_model
    if len(shapes)==1:  shapes = shapes+shapes
    if len(shapes)==2:  shapes = shapes+(len(channel_indx),)
    generator_model     = dict_to_model_parse( configs_dict[gen_choice],(shapes[0]//scale , shapes[1]//scale , shapes[2]) )
    discriminator_model = dict_to_model_parse( configs_dict[dis_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    content_model       = dict_to_model_parse( configs_dict[con_choice],(shapes[0]        , shapes[1]        , shapes[2]) )
    generator_model.name = 'gen_output'

    X_lr   = Input(shapes[0],name='lr_input')
    X_hr   = Input(shapes[1],name='hr_input')

    Y_sr   = generator(X_lr)    
    # Actual is given 1, fake is given 0
    Y_dis_sr  = discriminator(X_sr)
    Y_dis_hr  = discriminator(X_hr)
    Y_dis     = Concatenate(name='dis_output')([Y_dis_sr,Y_dis_hr])
    
    # RMSE of content layers defined below
    con_sr = content_model(X_sr)
    con_hr = content_model(X_hr)
    Y_con  = Subtract()([con_sr,con_hr])
    Y_con  = Lambda( lambda x : x**2 )                       (Y_con)
    Y_con  = Lambda( lambda x : x/memory_batch )             (Y_con)
    Y_con  = Lambda( lambda x : x**0.5 , name='con_output' ) (Y_con)
    
    return Model(inputs=[X_lr,X_hr],outputs=[Y_sr,Y_dis,Y_con],name=', '.join((gen_choice,dis_choice,con_choice)))

def freeze_model(model):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False

def unfreeze_model(model):
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True

def compile_model(model,mode,opt):
    if mode in ('cnn','gen'):
        train_model , non_train_model = generator_model, discriminator_model
        loss = { 'gen_output':'MSE' , 'dis_output':'GEN_LOSS' , 'con_output':'ZERO_LOSS' }
        if mode=='cnn':
            loss_weights = { 'gen_output':1 , 'dis_output':0 , 'con_output':0    }
        elif mode=='gen':
            loss_weights = { 'gen_output':0 , 'dis_output':1 , 'con_output':1e-3 }
    elif mode=='dis':
        non_train_model , train_model = generator_model, discriminator_model
        loss = { 'gen_output':'MSE' , 'dis_output':'DIS_LOSS' , 'con_output':'ZERO_LOSS' }
        loss_weights = { 'gen_output':0 , 'dis_output':1 , 'con_output':1e-3 }
    
    metrics = {'gen_output':'PSNR'}
    freeze_model(   content_model )
    freeze_model(   non_train_model )
    unfreeze_model( discriminator_model )
    model.compile(  optimizer=opt , loss = loss , loss_weights = loss_weights, metrics )
    freeze_model(   content_model )
    freeze_model(   generator_model )
    unfreeze_model( discriminator_model )

'''
gan_model = my_gan( block_size )
gan_model = compiled_model(gan_model,'cnn','Adam')


X_train = get_data(data_name,'train',indx=['LR','HR'],org=False)

train_input  = {
    'lr_input'  :X_train[0] ,
    'hr_input': X_train[1]
    }
train_output = {
    'gen_output':X_train[1] ,
    'dis_output':np.array([[0,1],]*len(X_train[0]),dtype=float_precision) ,
    'con_output':np.array([None]*len(X_train[0]))
    }

model.fit( train_input , train_output,epochs=inner_epochs, batch_size=memory_batch)
'''

######## MODELS DEFINITIONS ENDS ########


######## PLOTTING FUNCTIONS BEGINS HERE ########

def plot_history(history):
    plot_dir =  os.path.join('.','training_plots',gan_model.name)
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    all_keys      = set( history.keys() )
    non_val_keys  = set( x for x in all_keys if not x.startswith('val') )
    both_keys     = set( x for x in all_keys if (x in non_val_keys and 'val_'+x in all_keys) )
    non_val_keys  = both_keys - non_val_keys
    only_val_keys = all_keys - ( both_keys|non_val_keys )

    for ix,keys in enumerate((non_val_keys,only_val_keys,all_keys)):
        for key in keys:
            if ix==0:
                _ = plt.plot(history[key]         , linewidth=1 , label=key )
            if ix==1:
                _ = plt.plot(history[key]         , linewidth=1 , label=key )
            if ix==2:
                _ = plt.plot( history[key]        , linewidth=1 , label=key        )
                _ = plt.plot( history['val_'+key] , linewidth=1 , label='val_'+key )
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3e}')) # 2 decimal places
            plt.xticks( np.round(np.linspace(0,len(history[key])-1,10),2) , np.round(np.linspace(1,len(history[key]),10),2) )
            plt.grid(True) ; plt.xlabel('Epochs') ; plt.ylabel(key) ; plt.title(key.upper())
            plt.legend( loc='upper left' , bbox_to_anchor=(1,1) , fancybox=True , shadow=True )
            plt.savefig( os.path.join(plot_dir,'{}.png'.format(key)) , dpi=600 , bbox_inches='tight' , format='PNG' )
            plt.close()

######## PLOTTING FUNCTIONS ENDS HERE ########



######## LEARNING RATE SCHEDULES BEGINS HERE ########

# Presently Arihtmetic Progression based between [lf,flr], O_E*N_DB
pass

######## LEARNING RATE SCHEDULES BEGINS HERE ########


# From choices available for sub-models of GAN, both train & test needs this
gan_model = my_gan( block_size )


''' TRAINING CODE SECTION BEGINS '''

if train_flag:
    train(data_name,train_strategy,(1,1))

def train(name,train_strategy,dis_gen_ratio=(1,1)):
    ""
    global iSUP, ia, ib, gan_model
    iSUP, ia, ib = SUP, a, b
    # Modes in which have to train model, either CNN only interleaving using GAN
    modes = ['cnn'] if train_strategy=='cnn' else ['dis']*gen_dis_ratio[0]+['gen']**gen_dis_ratio[1]
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
    
    # Using previous model if so exist and set as command flag to use, warm training has many advantages
    gan_model_path = os.path.join(save_dir, gan_model.name)
    if os.path.isdir(save_dir) and gan_model.name in os.listdir(save_dir):
        if prev_model:
            gan_model = keras.models.load_model(gan_model_path,custom_objects=keras_update_dict)
        else:
            os.remove(gan_model_path)

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
    'gen_output':x_valid['hr_input'] ,
    'dis_output':np.array([[0,1],]*len(x_valid[0]),dtype=float_precision) ,
    'con_output':np.array([None]*len(x_valid[0]))
    }
    val_init_PSNR = get_IPSNR(x_valid['hr_input'],x_valid['lr_input'],pred_mode='LR')
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
            compile_model(gan_model,mode,opt)
            np.random.shuffle(file_names[name]['train'])
            iSUP = int(np.round(iSUP)) if int(np.round(iSUP))%2 else int(np.round(iSUP))+1
            
            for i in range(n_disk_batches):
                K.set_value(gan_model.optimizer.lr    , lr+(i+epc*n_disk_batches)*lr_delta)
                K.set_value(gan_model.optimizer.decay , decay)
                if gclip:
                    if gnclip is not None:
                        gan_model.optimizer.__dict__['clipnorm']  = gnclip / lr+(i+epc*n_disk_batches)*lr_delta
                    if gvclip is not None:
                        gan_model.optimizer.__dict__['clipvalue'] = gvclip / lr+(i+epc*n_disk_batches)*lr_delta
                
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
                'gen_output':x_train['hr_input'] ,
                'dis_output':np.array([[0,1],]*len(x_valid[0]),dtype=float_precision) ,
                'con_output':np.array([None]*len(x_valid[0]))
                }
                train_init_PSNR = get_IPSNR(x_train['hr_input'],x_train['lr_input'],pred_mode='LR')

                history = model.fit(x=x_train,y=y_train,epochs=inner_epochs,batch_size=memory_batch,validation_data=(x_valid,y_valid),shuffle=True,verbose=True)
                history = history.history

                del x_train, y_train

                history['IPSNR']     = [ (i-train_init_PSNR) for i in history['PSNR'] ]
                history['val_IPSNR'] = [ (i-val_init_PSNR)   for i in history['val_PSNR'] ]
                all_history = history if (all_history is None) else { i:(all_history[i]+history[i])for i in history }
                plot_history(all_history) # Plotting progress after each disk batch

                #Saving Trained model, after all QP values of currect disk_batch were processed
                print( '\nSaving Trained model > ',end='' )
                # Save model and weights
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                try:
                    gan_model.save(gan_model_path)
                    generator_model.save(os.path.join(save_dir,gen_choice))
                except:
                    print('Model Cannot be Saved')
                else:
                    print('Saved trained model at %s \n\n' % gan_model_path)

                del datasets[name]['train']
                iSUP, ia, ib = iSUP+SUP_delta, ia+a_delta, ib+b_delta
    del x_valid , y_valid, datasets['LR'][name]['valid'] , datasets['HR'][name]['valid']

''' TRAINING CODE SECTION ENDS '''


''' EVALUATING AND GENERATING MODEL OUTPUTS ON TEST BEGINS '''

if test_flag:
    test(data_name)

def test(name):
    generator_model_path = os.path.join(save_dir,gen_choice)
    file_names[name]['test'] = sorted(os.listdir(os.path.join(high_path,'test')))[::-1] #profiling

    # Working each image at a time for full generation
    disk_batch = 1 # Taking disk_batch to be 1 as heterogenous images
    n_disk_batches = len(file_names['test'][:test_images_limit])

    test_csv = open('{} {}.csv'.format(model_choice,n_disk_batches),'w')
    print( 'index,img_name,initial_psnr,final_psnr,psnr_gain' , file=test_csv )

    psnr_sum = {}
    init = datetime.now()
    if os.path.isdir(save_dir) and gen_choice in os.listdir(save_dir):
        generator_model = keras.models.load_model(generator_model_path,custom_objects=keras_update_dict)
    print('\nTime to Load Model',datetime.now()-init) # profiling

    init = datetime.now()
    new_generator_model = dict_to_model_parse( configs_dict[gen_choice] , (None,None,len(channel_indx)) )
    print('Time to Build New Model',datetime.now()-init) # profiling

    init = datetime.now()
    new_generator_model.set_weights(generator_model.get_weights())
    print('Time to Set Weights in New Model',datetime.now()-init) # profiling

    # Naming convention based name changing
    if imwrite:
        gen_store = os.path.join(gen_path,name)
        try:
            shutil.rmtree(gen_store)
        except:
            pass
        os.makedirs(gen_store)

    for i in range(n_disk_batches):
        init_time = datetime.now()
        feed_data(name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,False,'test',True) # erase all previous dataset
        print('\nTime to read image {} is {}'.format(i+1,datetime.now()-init_time))

        init = datetime.now()        
        x_test = get_data(name,'test',indx=['LR','HR'],org=False)
        print( 'Time to Scale Image',datetime.now()-init ) # profiling
        
        y_test = x_test[1]
        y_pred = new_generator_model.predict(x_test[0],verbose=1,batch_size=1)
        x_test = x_test[0]

        init = datetime.now()
        psnr_i =  get_IPSNR(np.array(y_test),np.array(x_test),pred_mode='LR')
        psnr_f =  get_IPSNR(np.array(y_test),np.array(x_pred),pred_mode='HR')
        psnr_g = psnr_f - psnr_i

        psnr_sum['initial'] = psnr_i if ('initial' not in psnr_sum) else (psnr_sum['initial']+psnr_i)
        psnr_sum['final'  ] = psnr_f if ('final'   not in psnr_sum) else (psnr_sum['final'  ]+psnr_f)
        psnr_sum['gain'   ] = psnr_g if ('gain'    not in psnr_sum) else (psnr_sum['gain'   ]+psnr_g)
        print('Time to Find and Store PSNRs',datetime.now()-init) # profiling

        print('Initial PSNR = {}, Final PSNR = {}, Gained PSNR = {}'.format(psnr_i,psnr_f,psnr_g))

        init = datetime.now()
        test_csv = open('{} {}.csv'.format(model_choice,n_disk_batches),'a')
        print( '{},{},{},{},{}\n'.format(i+1,file_names['test'][i],psnr_i,psnr_f,psnr_g) , file=test_csv )
        test_csv.close()
        print('Time to Update CSV file',datetime.now()-init) # profiling

        if imwrite and bit_depth==8:
            init = datetime.now()
            img = Image.fromarray( backconvert(y_pred[0],typ='HR') )
            img.save( os.path.join(gen_dir,'{}.PNG'.format(file_names['test'][0])) )
            print('Time to Write Image',datetime.now()-init) # profiling

        del x_test, y_test, y_pred, datasets[name]['test']

    # Finding average scored of learned model
    print('\nTest Statistics for QP = {}'.format(key))
    print('Unweighted Initial PSNR:', psnr_sum_i[key]/n_disk_batches )
    print('Unweighted Final   PSNR:', psnr_sum_f[key]/n_disk_batches )
    print('Unweighted Gain in PSNR:', psnr_sum_g[key]/n_disk_batches )

''' EVALUATING AND GENERATING MODEL OUTPUTS ON TEST ENDS '''



''' UNUSED CODE SECTION BEGINS '''

'''
'''
