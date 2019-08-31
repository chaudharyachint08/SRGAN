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
from keras.layers import Add, Subtract, Multiply, Average, Maximim, Minimum, Concatenate
# Advanced Activation Layers
advanced_activations = ('LeakyReLU','PReLU','ELU')
for i in advanced_activations:
    exec('from keras.layers import {}'.format(i))
# Normalization Layers
from keras.layers import BatchNormaliza1tion
# Vgg19 for Content loss applications
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
# Used for image.resize(target_size,PIL.Image.BICUBIC)
import PIL


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
parser.add_argument("--outer_epochs",       type=eval, dest='outer_epochs',        default=5)
parser.add_argument("--inner_epochs",       type=eval, dest='inner_epochs',        default=2)
parser.add_argument("--disk_batches_limit", type=eval, dest='disk_batches_limit',  default=None)
parser.add_argument("--valid_images_limit", type=eval, dest='valid_images_limit',  default=None)
parser.add_argument("--test_images_limit",  type=eval, dest='test_images_limit',   default=None)
parser.add_argument("--seed",               type=eval, dest='np_seed',             default=None)

parser.add_argument("--disk_batch",         type=eval, dest='disk_batch',          default=20)
parser.add_argument("--memory_batch",       type=eval, dest='memory_batch',        default=128)
parser.add_argument("--block_size",         type=eval, dest='block_size',          default=32)
parser.add_argument("--bit_depth",          type=eval, dest='bit_depth',           default=8)
parser.add_argument("--min_q",              type=eval, dest='min_q',               default=0)
parser.add_argument("--max_q",              type=eval, dest='max_q',               default=51)
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
parser.add_argument("--model_choice", type=str, dest='model_choice',    default='baseline')
parser.add_argument("--loss",         type=str, dest='loss',            default='mean_squared_error')
parser.add_argument("--attention",    type=str, dest='attention',       default='sigmoid')
parser.add_argument("--precision",    type=str, dest='float_precision', default='float32')
parser.add_argument("--optimizer",    type=str, dest='optimizer',       default='Adam')
parser.add_argument("--name",         type=str, dest='data_name',       default='CAR')
parser.add_argument("--save_dir",     type=str, dest='save_dir',        default='saved_CAR_models')
parser.add_argument("--high_path",    type=str, dest='high_path',       default='/vol1/dbstore/orc_srib/amith.ds/CLIC/Dataset')
parser.add_argument("--low_path",     type=str, dest='low_path',        default='/vol1/dbstore/orc_srib/amith.ds/CLIC_ENCODE')

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



######## METRIC DEFINITIONS BEGINS ########

def PSNR(y_true, y_pred,round_flg=True):
    if round_flg:
        y_true, y_pred = K.clip( K.round(y_true*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) ), K.clip( K.round(y_pred*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
    else:
        y_true, y_pred = K.clip( (y_true*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) ), K.clip( (y_pred*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
    return (10.0 * K.log((((1<<bit_depth)-1) ** 2) / (K.mean(K.square(y_pred - y_true))))) / K.log(10.0)

def npPSNR(y_true, y_pred,round_flg=True):
    if round_flg:
        y_true = np.clip( np.round(y_true*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
        y_pred = np.clip( np.round(y_pred*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
    else:
        y_true = np.clip( (y_true*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
        y_pred = np.clip( (y_pred*((1<<bit_depth)-1)), 0, ((1<<bit_depth)-1) )
    return (10.0 * np.log((((1<<bit_depth)-1) ** 2) / (np.mean(np.square(y_pred - y_true))))) / np.log(10.0)

######## METRIC DEFINITIONS ENDS ########



######## CUSTOM LOSS FUNCTIONS DEFINITIONS BEGINS ########

def fgauss(size,sigma):
  x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
  g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
  return g/g.sum()

def Kfgauss(size,sigma,channels):
  fg = np.array([fgauss(size,sigma)]*channels).T[:,:,:,np.newaxis]
  return K.constant( fg )

def MIX( X,Y,a=a,M=SUP, C1=C1,C2=C2, alpha=alpha,beta=beta ):
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


custom_losses = ('MIX',)

######## CUSTOM LOSS FUNCTIONS DEFINITIONS ENDS ########



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

def mymodel(config,*ip_shapes):
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

######## MODELS DEFINITIONS ENDS ########



######## DATA STORAGE, READING & PROCESSING FUNCTIONS BEGINS ########

datasets = {} # Global Dataset Storage
file_names = {} # Global filenames for disk_batching

def normalizeC(mat):
    if str(mat.dtype)=='object':
        val = np.array([ (i.astype(float_precision)/((1<<bit_depth)-1)) for i in mat])
    else:
        val = mat.astype(float_precision) / ((1<<bit_depth)-1)
    return val

def normalizeQ(mat):
    if type(mat) in (int,float):
        val = (mat-min_q)/(max_q-min_q)
    elif str(mat.dtype)=='object':
        val = np.array([((i.astype(float_precision)-min_q)/(max_q-min_q)) for i in mat])
    else:
        val = (mat.astype(float_precision)-min_q) / (max_q-min_q)
    return val

def backconvert(mat):
    mat = mat*((1<<bit_depth)-1)
    return np.clip( mat.round(), 0, ((1<<bit_depth)-1) )

def prepare_dataset(IMG,QP=32,channel_indx=(0,)):
    norm_IMG = normalizeC(np.array([ i[:,:,channel_indx] for i in IMG ]))
    norm_QP  = normalizeQ(QP)
    if str(norm_IMG.dtype)=='object':
        val = [ norm_IMG, np.array([np.ones(i.shape[:-1]+(1,),dtype=float_precision)*norm_QP for i in norm_IMG]) ]
    else:
        val = [ norm_IMG, np.ones(norm_IMG.shape[:-1]+(1,),dtype=float_precision)*norm_QP ]
    return val

def on_fly_crop(mat,block_size=block_size,overlap=overlap):
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
    
def feed_data(name, low_path, high_path, lw_indx, up_indx, crop_flag = True, typ = 'train'):
    if name not in datasets:
        datasets[name] = {}
    for phase in ('train','valid','test'): # All datasets be present, even if empty
        if phase not in datasets[name]:
            datasets[name][phase] = {}
    # Feeding Ground Truth (High Quality Images)
    datasets[name][typ]['high'] = []
    # os.listdir(os.path.join(high_path,typ))
    for img_name in file_names[typ][lw_indx:up_indx]:
        if read_as_gray:
            img_mat = np.round(((1<<bit_depth)-1)*io.imread(os.path.join(high_path,typ,img_name),as_gray=read_as_gray))[:,:,np.newaxis]
        else:
            img_mat = io.imread(os.path.join(high_path,typ,img_name),as_gray=read_as_gray)
        if crop_flag:
            datasets[name][typ]['high'].extend(on_fly_crop(img_mat))
        else:
            datasets[name][typ]['high'].append(img_mat)
    # Feeding Encoded Images (Low Quality Images)
    for fldr_name in os.listdir(low_path):
        if ('qp' in fldr_name) and os.path.isdir(os.path.join(low_path,fldr_name)):
            q_value = fldr_name.split('_')[-1].strip()
            datasets[name][typ][q_value] = []
            for img_name in file_names[typ][lw_indx:up_indx]:
                if read_as_gray:
                    img_mat = np.round(((1<<bit_depth)-1)*io.imread(os.path.join(low_path,fldr_name,'output',typ,img_name),as_gray=read_as_gray))[:,:,np.newaxis]
                else:
                    img_mat = io.imread(os.path.join(low_path,fldr_name,'output',typ,img_name),as_gray=read_as_gray)
                if crop_flag:
                    datasets[name][typ][q_value].extend(on_fly_crop(img_mat))
                else:
                    datasets[name][typ][q_value].append(img_mat)
    #Converting into Numpy Arrays
    for typ in datasets[name]:
        for key in list(datasets[name][typ].keys()):
            datasets[name][typ][key] = np.array([i.astype('uint8') for i in datasets[name][typ][key]])

def get_data(typ,key,indx=(0,1),org=False): # Flag to get as uint8 or float values
    global q_value
    if 0 in indx:
        q_value = int(key)
    if 0 in indx: x = datasets[data_name][typ][key]
    if 1 in indx: y = datasets[data_name][typ]['high']
    if not org:
        if 0 in indx: x = prepare_dataset(x,q_value,channel_indx)
        if 1 in indx: y = normalizeC(y)
    return (x, y) if indx==(0,1) else (x if indx==(0,) else y)

######## DATA STORAGE, READING & PROCESSING FUNCTIONS ENDS ########



######## PLOTTING FUNCTIONS BEGINS HERE ########

def plot_history(history):
    plot_dir =  os.path.join('training_plots','plots - '+model.name)
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    flag = False ; plt.close()
    for key in history:
        if 'loss' in key:
            flag = True
            _ = plt.plot(history[key],linewidth=1,label=key)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3e}')) # 2 decimal places
    if flag:
        plt.xticks( np.round(np.linspace(0,len(history[key])-1,10),2) , np.round(np.linspace(1,len(history[key]),10),2) )
        plt.grid(True)
        plt.xlabel('Epochs') ; plt.ylabel(loss) ; plt.title('Training Loss')
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),fancybox=True,shadow=True)
        plt.savefig(os.path.join(plot_dir,'Training Loss.png'),dpi=600,bbox_inches='tight',format='PNG')
        plt.close()

    flag = False ; plt.close()
    for key in history:
        if 'PSNR' in key and 'IPSNR' not in key:
            flag = True
            _ = plt.plot(history[key],linewidth=1,label=key)
    if flag:
        plt.xticks( np.round(np.linspace(0,len(history[key])-1,10),2) , np.round(np.linspace(1,len(history[key]),10),2) )
        plt.grid(True)
        plt.xlabel('Epochs') ; plt.ylabel('PSNR') ; plt.title('Training PSNR')
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),fancybox=True,shadow=True)
        plt.savefig(os.path.join(plot_dir,'Training PSNR.png'),dpi=600,bbox_inches='tight',format='PNG')
        plt.close()

    flag = False ; plt.close()
    for key in history:
        if 'IPSNR' in key:
            flag = True
            _ = plt.plot(history[key],linewidth=1,label=key)
    if flag:
        plt.xticks( np.round(np.linspace(0,len(history[key])-1,10),2) , np.round(np.linspace(1,len(history[key]),10),2) )
        plt.grid(True)
        plt.xlabel('Epochs') ; plt.ylabel('Inc-PSNR') ; plt.title('Training Inc-PSNR')
        plt.legend(loc='upper left',bbox_to_anchor=(1,1),fancybox=True,shadow=True)
        plt.savefig(os.path.join(plot_dir,'Training Inc-PSNR.png'),dpi=600,bbox_inches='tight',format='PNG')
        plt.close()

######## PLOTTING FUNCTIONS ENDS HERE ########



######## LEARNING RATE SCHEDULES BEGINS HERE ########

# Presently Arihtmetic Progression based between [lf,flr], O_E*N_DB
pass

######## LEARNING RATE SCHEDULES BEGINS HERE ########



''' TRAINING CODE SECTION BEGINS '''

if train_flag:
    file_names['train'] = sorted(os.listdir(os.path.join(high_path,'train')))
    file_names['valid'] = sorted(os.listdir(os.path.join(high_path,'valid')))

    n_disk_batches = disk_batches_limit if (disk_batches_limit is not None) else int(np.ceil(len(file_names['train'])/disk_batch))

    print('\nOuter Epochs {} Disk Batches {}'.format(outer_epochs,n_disk_batches))
    # Building Model
    if custom_model:
        model = mymodel( configs_dict[model_choice], (block_size,block_size,len(channel_indx)),(block_size,block_size,1) ) # [(m*b*b*c),(m*b*b*1)]
    else:
        model = baseline( (block_size,block_size,len(channel_indx)),(block_size,block_size,1) ) # [(m*b*b*c),(m*b*b*1)]

    if   optimizer == 'Adam':
        opt = keras.optimizers.Adam( lr=lr, decay=decay )
    elif optimizer == 'SGD':
        opt = keras.optimizers.SGD( lr=lr, decay=decay, momentum=momentum )
    else:
        raise Exception('Unexpected Optimizer than two classics')

    if loss in custom_losses:
      model.compile(optimizer=opt, loss=eval(loss), metrics=[PSNR])
    else:
      model.compile(optimizer=opt, loss=loss, metrics=[PSNR])
    model_path = os.path.join(save_dir, model.name)

    if os.path.isdir(save_dir) and model.name in os.listdir(save_dir):
        if prev_model:
            model = keras.models.load_model(model_path,custom_objects={'PSNR':PSNR,'MIX':MIX})
        else:
            os.remove(model_path)

    lr_delta  = 0 if (flr  is None) else ((flr-lr)   / (n_disk_batches*outer_epochs-1))
    a_delta   = 0 if (fa   is None) else ((fa-a)     / (n_disk_batches*outer_epochs-1))
    b_delta   = 0 if (fb   is None) else ((fb-b)     / (n_disk_batches*outer_epochs-1))
    SUP_delta = 0 if (fSUP is None) else ((fSUP-SUP) / (n_disk_batches*outer_epochs-1))

    K.set_value(model.optimizer.lr,    lr)
    K.set_value(model.optimizer.decay, decay)
    
    global_history = None

    #Reading Validation Set
    np.random.shuffle(file_names['valid'])
    print('\nReading Validation Set',end=' ') ; init_time = datetime.now()
    feed_data(data_name,low_path,high_path,None,valid_images_limit,True,'valid')
    print( 'Time taken is {} '.format((datetime.now()-init_time)) )

    x_valid, y_valid = None, None
    for key in sorted(datasets[data_name]['valid']):
        if key.isnumeric():
            _1, _2 = get_data('valid',key)
            x_valid = _1 if (x_valid is None) else [ np.concatenate( (x_valid[0], _1[0]) ), np.concatenate( (x_valid[1], _1[1]) ) ]
            y_valid = _2 if (y_valid is None) else np.concatenate( (y_valid, _2) )

    val_init_PSNR = npPSNR(y_valid,x_valid[0])
    iSUP = SUP

    for epc in range(outer_epochs):
        np.random.shuffle(file_names['train'])

        SUP = int(np.round(iSUP)) if int(np.round(iSUP))%2 else int(np.round(iSUP))+1

        print('\nOuter Epoch {}'.format(epc+1))
        for i in range(n_disk_batches):
            K.set_value(model.optimizer.lr,    lr+(i+epc*n_disk_batches)*lr_delta)
            if gclip:
                if gnclip is not None:
                    model.optimizer.__dict__['clipnorm']  = gnclip / lr+(i+epc*n_disk_batches)*lr_delta
                if gvclip is not None:
                    model.optimizer.__dict__['clipvalue'] = gvclip / lr+(i+epc*n_disk_batches)*lr_delta
            
            print('Reading Disk Batch {}'.format(i+1),end='') ; init_time = datetime.now()
            feed_data(data_name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,True,'train')
            print(' Time taken is {} '.format((datetime.now()-init_time)),end='')

            if mix_diff_qps:
                x_train, y_train = None, None
                print('\n > Scaling Batch of all QP')
                for key in sorted(datasets[data_name]['train']):                
                    if key.isnumeric():
                        _1, _2 = get_data('train',key)
                        x_train = _1 if (x_train is None) else [ np.concatenate( (x_train[0], _1[0]) ), np.concatenate( (x_train[1], _1[1]) ) ]
                        y_train = _2 if (y_train is None) else np.concatenate( (y_train, _2) )
                history = model.fit(x=x_train,y=y_train,epochs=inner_epochs,batch_size=memory_batch,validation_data=(x_valid,y_valid),shuffle=True,verbose=True)
                history = history.history
                init_PSNR = npPSNR(y_train, x_train[0])
                del x_train, y_train
                history['IPSNR']     = [(i-init_PSNR)     for i in history['PSNR']]
                history['val_IPSNR'] = [(i-val_init_PSNR) for i in history['val_PSNR']]
                global_history = history if (global_history is None) else { i:(global_history[i]+history[i])for i in history }
            else:
                y_train = get_data('train',None,(1,))
                for key in sorted(datasets[data_name]['train']):
                    if key.isnumeric():
                        print('\n > Scaling Batch {}'.format(key))
                        x_train = get_data('train',key,(0,))
                        history = model.fit(x=x_train,y=y_train,epochs=inner_epochs,batch_size=memory_batch,validation_data=(x_valid,y_valid),shuffle=True,verbose=True)
                        history = history.history
                        init_PSNR = npPSNR(y_train, x_train[0])
                        del x_train
                        history['IPSNR']     = [(i-init_PSNR)     for i in history['PSNR']]
                        history['val_IPSNR'] = [(i-val_init_PSNR) for i in history['val_PSNR']]
                        global_history = history if (global_history is None) else { i:(global_history[i]+history[i])for i in history }
                del y_train

            plot_history(global_history) # Plotting progress after each disk batch

            #Saving Trained model, after all QP values of currect disk_batch were processed
            print( '\nSaving Trained model > ',end='' )
            # Save model and weights
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            try:
                model.save(model_path)
            except:
                print('Model Cannot be Saved')
            else:
                print('Saved trained model at %s \n\n' % model_path)

            del datasets[data_name]['train']

            a, iSUP = a+a_delta, iSUP+SUP_delta
            
    del x_valid, y_valid

''' TRAINING CODE SECTION ENDS '''



''' EVALUATING AND GENERATING MODEL OUTPUTS ON TEST BEGINS '''

if test_flag:
    model_path = os.path.join(save_dir,(model.name if train_flag else model_choice))

    file_names['test'] = sorted(os.listdir(os.path.join(high_path,'test')))[::-1] #profiling

    # Working each image at a time for full generation
    disk_batch = 1 # Taking disk_batch to be 1 as heterogenous images
    n_disk_batches = len(file_names['test'][:test_images_limit])

    # Deleting Validation set
    if (data_name in datasets) and ('valid' in datasets[data_name]):
        del datasets[data_name]['valid']

    test_csv = open('{} {}.csv'.format(model_choice,n_disk_batches),'w')
    test_csv.write( 'index,file_name,initial_psnr,final_psnr,psnr_gain\n' )

    psnr_sum_i, psnr_sum_f, psnr_sum_g = {}, {}, {}

    init = datetime.now()
    if os.path.isdir(save_dir) and (model.name if train_flag else model_choice) in os.listdir(save_dir):
        model = keras.models.load_model(model_path,custom_objects={'PSNR':PSNR,'MIX':MIX})
    print('\nTime to Load Model',datetime.now()-init) # profiling

    init = datetime.now()
    if custom_model:
        new_model = mymodel(configs_dict[model_choice], (None,None,len(channel_indx)), (None,None,1) )
    else:
        new_model = baseline( (None,None,len(channel_indx)), (None,None,1) )
    print('Time to Build New Model',datetime.now()-init) # profiling

    init = datetime.now()
    new_model.set_weights(model.get_weights())
    print('Time to Set Weights in New Model',datetime.now()-init) # profiling

    for i in range(n_disk_batches):
        init_time = datetime.now()
        feed_data(data_name,low_path,high_path,i*disk_batch,(i+1)*disk_batch,False,'test')
        print('\nTime to read image {} is {}'.format(i+1,datetime.now()-init_time))

        if imwrite and (not i):
            for key in datasets[data_name]['test']:
                if key.isnumeric():
                    gen_dir = 'Gen_Images {} - {}'.format(model.name,key)
                    try:
                        shutil.rmtree(gen_dir)
                    except:
                        pass
                    os.mkdir(gen_dir)

        y_test = get_data('test',None,(1,))
        for key in sorted(datasets[data_name]['test']):
            if key.isnumeric():
                gen_dir = 'Gen_Images {} - {}'.format(model.name,key)

                init = datetime.now()
                x_test = get_data('test',key,(0,))
                print('Time to Scale Image',datetime.now()-init,end=' ') # profiling

                y_pred = new_model.predict(x_test,verbose=1,batch_size=1)

                init = datetime.now()
                psnr_i = npPSNR(y_test,x_test[0])
                psnr_f = npPSNR(y_test,y_pred)
                psnr_g = psnr_f - psnr_i
                psnr_sum_i[key] = psnr_i if (key not in psnr_sum_i) else (psnr_sum_i[key]+psnr_i)
                psnr_sum_f[key] = psnr_f if (key not in psnr_sum_f) else (psnr_sum_f[key]+psnr_f)
                psnr_sum_g[key] = psnr_g if (key not in psnr_sum_g) else (psnr_sum_g[key]+psnr_g)
                print('Time to Find and Store PSNRs',datetime.now()-init) # profiling

                print('Initial PSNR = {}, Final PSNR = {}, Gained PSNR = {}'.format(psnr_i,psnr_f,psnr_g))

                init = datetime.now()
                test_csv = open('{} {}.csv'.format(model_choice,n_disk_batches),'a')
                test_csv.write( '{},{},{},{},{}\n'.format(i+1,file_names['test'][i],psnr_i,psnr_f,psnr_g) )
                test_csv.close()
                print('Time to Update CSV file',datetime.now()-init) # profiling

                if imwrite:
                    init = datetime.now()
                    cv2.imwrite(os.path.join(gen_dir,file_names['test'][i]+' A.png'),cv2.cvtColor(backconvert(x_test[0][0]), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(gen_dir,file_names['test'][i]+' B.png'),cv2.cvtColor(backconvert(y_pred[0]),    cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(gen_dir,file_names['test'][i]+' C.png'),cv2.cvtColor(backconvert(y_test[0]),    cv2.COLOR_RGB2BGR))
                    print('Time to Write 3 Images',datetime.now()-init) # profiling

                del x_test

        del y_test, datasets[data_name]['test']

    # Finding average scored of learned model
    for key in psnr_sum_i:
        print('\nTest Statistics for QP = {}'.format(key))
        print('Unweighted Initial PSNR:', psnr_sum_i[key]/n_disk_batches )
        print('Unweighted Final   PSNR:', psnr_sum_f[key]/n_disk_batches )
        print('Unweighted Gain in PSNR:', psnr_sum_g[key]/n_disk_batches )

''' EVALUATING AND GENERATING MODEL OUTPUTS ON TEST ENDS '''



''' UNUSED CODE SECTION BEGINS '''

'''
'''
