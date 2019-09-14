'''
File for defining Model & Block definitions,
Note: Models are constrained & have their own indexing mechanism
While there is no indexing mechanism withing Blocks
Block are simply n-ary input & single tensor output functions
'''

def initiate(dct):
    global configs_dict    
    channel_indx = dct['channel_indx']

    for i in dct:        
        try:
            exec( 'global {}; {} = dct[{}]'.format(i,i,repr(i)) )
        except: # If any variable cannot be assigned as it is 
            pass
    try:
        for i in dct['advanced_activations']:
            try:
                exec('{}=dct[{}]'.format(i,repr(i)))
            except: # If any specific advanced activation is not found
                pass
    except: # If 'advanced_activations' is not found
        pass


    def gen_residual_block(X_input):
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X = BatchNormalization()(X)
        X = PReLU(shared_axes=(1,2))(X)
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X)
        X = BatchNormalization()(X)
        X = Add()([X,X_input[0]])
        return X

    def gen_last_block(X_input):
        X = Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        # X = UpSampling2D(  size=2,interpolation=upsample_interpolation) (X)
        X = PixelShuffle(size=2)(X)
        X = PReLU(shared_axes=(1,2))(X)
        return X

    baseline_gen = {
        'name'      :'baseline_gen',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+7,B+8),],
        'convo_sub':['Conv2D','Conv2D','Conv2D',],
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)          for i in range(B)] + [(B+5+i,B+5+i+1)  for i in range(2)],
        'block_sub':['gen_residual_block' for i in range(B)] + ['gen_last_block' for i in range(2)],
        # BatchNormalization Links
        'btnrm'    :[(B+3,B+4),],
        # Merge Links
        'merge'    :[(2,B+4,B+5),],
        'merge_sub':['Add'],
        }

    baseline_dis = {
        'name'      :'baseline_dis',
        # Convolution Links
        'convo'    :[(0,1),(2,3),(5,6),(8,9),(11,12),(14,15),(17,18),(20,21),],
        'convo_sub':['Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'convo_par':[{'filters':64 ,'kernel_size':(3,3)}, {'filters':64 ,'kernel_size':(3,3),'strides':(2,2)},
                     {'filters':128,'kernel_size':(3,3)}, {'filters':128,'kernel_size':(3,3),'strides':(2,2)},
                     {'filters':256,'kernel_size':(3,3)}, {'filters':256,'kernel_size':(3,3),'strides':(2,2)},
                     {'filters':512,'kernel_size':(3,3)}, {'filters':512,'kernel_size':(3,3),'strides':(2,2)},],
        # Advanced Activation Links
        'aactv'    :[(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(19,20),(22,23),(25,26),],
        'aactv_sub':['LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU',],
        # BatchNormalization Links
        'btnrm'    :[(3,4),(6,7),(9,10),(12,13),(15,16),(18,19),(21,22),],
        # Flatten Links
        'flttn'    :[(23,24),],
        # Dense Links
        'dense'    :[(24,25),(26,27),],
        'dense_par':[{'units':1024},{'units':1},],
        # Activation Links
        'actvn'    :[(27,28),],
        'actvn_par':[{'activation':repr('sigmoid')},],
        }

    baseline_con = {
        'name'     :'baseline_con',
        'clpre'    :['vgg19',],
        'clpre_sub':['VGG19',],
        'clpre_out':['block5_conv4'], # Which layer to use
        'clpre_act':[True],           # If to use last activation or now
        }


    def gen_residual_block2(X_input):
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(X_input[0])
        # X = BatchNormalization()(X)
        X = PReLU(shared_axes=(1,2))(X)
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(X)
        # X = BatchNormalization()(X)
        X = Add()([X,X_input[0]])
        return X


    baseline_gen2 = {
        'name'      :'baseline_gen2',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+7,B+8),(0,B+9)],
        'convo_sub':['Conv2D','Conv2D','Conv2D','UpSampling2D',],
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same')},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same')},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same')},
                     {'size':scale ,'interpolation':repr(upsample_interpolation)},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)          for i in range(B)] + [(B+5+i,B+5+i+1)  for i in range(2)],
        'block_sub':['gen_residual_block2' for i in range(B)] + ['gen_last_block' for i in range(2)],
        # BatchNormalization Links
        'btnrm'    :[(B+3,B+4),],
        # Merge Links
        'merge'    :[(2,B+4,B+5),(B+8,B+9,B+10)],
        'merge_sub':['Add','Add',],
        }


    configs_dict = {

        #  SR-GAN ARCHITECTURE STORAGE
        'gen_residual_block'  : gen_residual_block,
        'gen_last_block'      : gen_last_block,
        'baseline_gen'        : baseline_gen ,
        'baseline_dis'        : baseline_dis ,
        'baseline_con'        : baseline_con ,

        'gen_residual_block2' : gen_residual_block2,
        'baseline_gen2'       : baseline_gen2 ,

        # E-SR-GAN ARCHITECTURE STORAGE


                    }









'''
    SRCNN_L8 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'SRCNN_L8',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11),(11,12),],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 32,64,64,64,64,64,128,len(channel_indx),],
        'l_kernel_size'    : [ (11,11),(3,3),(3,3),(3,3),(1,1),(5,5),(1,1),(5,5),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,12,13),(3,5,6),(3,8,9),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate','Concatenate',],
        }

    SRCNN_L4 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'SRCNN_L4',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(5,6),],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 48,64,64,len(channel_indx),],
        'l_kernel_size'    : [ (11,11),(3,3),(3,3),(5,5),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,6,7),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add',],
        }



    def rec_block(X_input):
        "blocks in general will be getting list of input"
        X_first = Conv2D(64,(3,3),activation='relu',padding='same')(X_input[0])
        batchnorm1, batchnorm2   = BatchNormalization(), BatchNormalization()
        activation1, activation2 = Activation('relu'),   Activation('relu')
        conv1                    = Conv2D(64,(3,3),activation='linear',padding='same')
        conv2                    = Conv2D(64,(3,3),activation='linear',padding='same')
        X = X_first
        for i in range(U):
            X = conv1(activation1(batchnorm1(X)))
            X = conv2(activation2(batchnorm2(X)))
            X = Add()([X,X_first])
        X_last = X
        return X_last

    drrn = {
        'name' : 'drrn',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [(B+2,B+3),],
        'l_conv_type'      : ['Conv2D',],
        'l_filters'        : [len(channel_indx),],
        'l_kernel_size'    : [(3,3),],
        'l_strides'        : [(1,1),],
        'l_padding'        : ['same',],
        'l_dilation_rate'  : [(1,1),],
        'l_activation'     : ['linear',],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,B+3,B+4),],
        'l_merge_type'   : ['Concatenate','Add',],
        # Blocks be defined here
        'l_block_connection' : [(i+2,i+3) for i in range(B)],
        'l_block_fun' : ['rec_block' for i in range(B)],
        }


'''