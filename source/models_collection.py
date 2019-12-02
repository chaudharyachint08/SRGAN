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



    trivial_con = {
        'name'      :'trivial_con' ,
        }

    
    trivial_dis = {
        'name'      :'trivial_dis' ,
        # Flatten Links
        'flttn'    :[(0,1),] ,
        # Dense Links
        'dense'    :[(1,2),] ,
        'dense_par':[{'units':1},] ,
        # Activation Links
        'actvn'    :[(1,2),] ,
        'actvn_par':[{'activation':repr('sigmoid')},] ,
        }


    baseline_con = {
        'name'     :'baseline_con',
        'clpre'    :['vgg19',],
        'clpre_sub':['VGG19',],
        'clpre_out':['block5_conv4'], # Which layer to use
        'clpre_act':[True],           # If to use last activation or now
        }




    def vdsr_block(X_input):
        X = Conv2D(filters=64               ,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_regularizer=keras.regularizers.l2(0.0001))(X_input[0])
        X = ReLU()(X)
        return X

    def vdsr_last_block(X_input):
        X = Conv2D(filters=len(channel_indx),kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_regularizer=keras.regularizers.l2(0.0001))(X_input[0])
        X = ReLU()(X)
        return X

    VDSR = { 'name'      :'VDSR' ,
        # Block Links
        'block'    :[(i,i+1)      for i in range(B)] + [(B,B+1),] ,
        'block_sub':['vdsr_block' for i in range(B)] + ['vdsr_last_block',] ,
        # Merge Links
        'merge'    :[(0,B+1,B+2),],
        'merge_sub':['Add'],
        }


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
        X = UpSampling2D(  size=2,interpolation=upsample_interpolation) (X)
        # X = PixelShuffle(size=2)(X)
        X = PReLU(shared_axes=(1,2))(X)
        return X

    SRResNet = { 'name'      :'SRResNet',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+5+int(np.ceil(np.log2(scale))),B+5+int(np.ceil(np.log2(scale)))+1),],
        'convo_sub':['Conv2D','Conv2D','Conv2D',],
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)          for i in range(B)] + [(B+5+i,B+5+i+1)  for i in range(int(np.ceil(np.log2(scale)))) ],
        'block_sub':['gen_residual_block' for i in range(B)] + ['gen_last_block' for i in range(int(np.ceil(np.log2(scale)))) ],
        # BatchNormalization Links
        'btnrm'    :[(B+3,B+4),],
        # Merge Links
        'merge'    :[(2,B+4,B+5),],
        'merge_sub':['Add'],
        }



    def ps_gen_last_block(X_input):
        X = Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        # X = UpSampling2D(  size=2,interpolation=upsample_interpolation) (X)
        X = PixelShuffle(size=2)(X)
        X = PReLU(shared_axes=(1,2))(X)
        return X

    PS_SRResNet = {
        'name'      :'PS_SRResNet',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+5+int(np.ceil(np.log2(scale))),B+5+int(np.ceil(np.log2(scale)))+1),],
        'convo_sub':['Conv2D','Conv2D','Conv2D',],
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)          for i in range(B)] + [(B+5+i,B+5+i+1)  for i in range(int(np.ceil(np.log2(scale)))) ],
        'block_sub':['gen_residual_block' for i in range(B)] + ['ps_gen_last_block' for i in range(int(np.ceil(np.log2(scale)))) ],
        # BatchNormalization Links
        'btnrm'    :[(B+3,B+4),],
        # Merge Links
        'merge'    :[(2,B+4,B+5),],
        'merge_sub':['Add'],
        }




    SRGAN_dis = {
        'name'      :'SRGAN_dis',
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




    def gen_residual_block2(X_input):
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(X_input[0])
        # X = BatchNormalization()(X)
        X = PReLU(shared_axes=(1,2))(X)
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(X)
        # X = BatchNormalization()(X)
        X = Lambda(lambda x:x*residual_scale)(X)
        X = Add()([X,X_input[0]])
        return X

    SRResNet2 = {
        'name'      :'SRResNet2',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+4+int(np.ceil(np.log2(scale))),B+4+int(np.ceil(np.log2(scale)))+1),(0,B+4+int(np.ceil(np.log2(scale)))+2)],
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
        'block'    :[(2+i,2+i+1)           for i in range(B)] + [(B+4+i,B+4+i+1)  for i in range(int(np.ceil(np.log2(scale)))) ],
        'block_sub':['gen_residual_block2' for i in range(B)] + ['gen_last_block' for i in range(int(np.ceil(np.log2(scale)))) ],
        # Merge Links
        'merge'    :[(2,B+3,B+4),(B+4+int(np.ceil(np.log2(scale)))+1,B+4+int(np.ceil(np.log2(scale)))+2,B+4+int(np.ceil(np.log2(scale)))+3)],
        'merge_sub':['Add','Add',],
        }




    def edsr_residual_block(X_input):
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X = PReLU(shared_axes=(1,2))(X)
        X = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X)
        X = Lambda(lambda x:x*residual_scale)(X)
        X = Add()([X,X_input[0]])
        return X

    def edsr_last_block(X_input):
        X = Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X = UpSampling2D(  size=2,interpolation=upsample_interpolation) (X)
        # X = PixelShuffle(size=2)(X)
        X = PReLU(shared_axes=(1,2))(X)
        return X

    EDSR = {
        'name'      :'EDSR',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+4+int(np.ceil(np.log2(scale))),B+4+int(np.ceil(np.log2(scale)))+1),],
        'convo_sub':['Conv2D','Conv2D','Conv2D',],
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)           for i in range(B)] + [(B+4+i,B+4+i+1)  for i in range(int(np.ceil(np.log2(scale)))) ],
        'block_sub':['edsr_residual_block' for i in range(B)] + ['gen_last_block' for i in range(int(np.ceil(np.log2(scale)))) ],
        # Merge Links
        'merge'    :[(2,B+3,B+4),],
        'merge_sub':['Add'],
        }




    def drrn_residual_block(X_input):
        "blocks in general will be getting list of input"
        X_first = Conv2D(64,(3,3),activation='relu',padding='same')(X_input[0])
        batchnorm1, batchnorm2   = BatchNormalization()     , BatchNormalization()
        activation1, activation2 = PReLU(shared_axes=[1,2]) , PReLU(shared_axes=[1,2])
        conv1                    = Conv2D(64,(3,3),activation='linear',padding='same')
        conv2                    = Conv2D(64,(3,3),activation='linear',padding='same')
        X = X_first
        for i in range(U):
            X = conv1(activation1(batchnorm1(X)))
            X = conv2(activation2(batchnorm2(X)))
            X = Add()([X,X_first])
        return X

    DRRN = {
        'name'      :'DRRN',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),] ,
        'convo_sub':['Conv2D','Conv2D',] ,
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)           for i in range(B)] ,
        'block_sub':['drrn_residual_block' for i in range(B)] ,
        # Merge Links
        'merge'    :[(0,B+3,B+4),],
        'merge_sub':['Add'],
        }




    def oam_residual_block(X_input):
        X1 = Conv2D(filters=12,kernel_size=(5,1),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X2 = Conv2D(filters=40,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X3 = Conv2D(filters=12,kernel_size=(1,5),strides=(1,1),padding='same',use_bias=True)(X_input[0])
        X  = Concatenate()([X1, X2, X3])
        X2 = GlobalAveragePooling2D()(X)
        X2 = Dense(128)(X2)
        X2 = PReLU()(X2)
        X2 = Dense(64)(X2)
        X2 = Activation(attention)(X2)

        batch_size      = X.get_shape().as_list()[ 0]
        number_channels = X.get_shape().as_list()[-1]
        # print(type(batch_size), type(number_channels))
        # print(X.shape, X2.shape)

        # This works well in NumPy some resheping issue need to be fixed
        # ((B.reshape((batch_size,1,1,number_channels))).T*A.T).T
        # Below Code is for testing re-shape & transposing, else code below is actual
        X2 = Reshape((1,1,number_channels))(X2)
        # print(X.shape, X2.shape)
        X  = Lambda(  lambda x:K.transpose( (K.transpose(x[1])*K.transpose(x[0])) )  )([X,X2])
        # print(X.shape)
        X = PReLU(shared_axes=(1,2))(X)
        X = Conv2D(filters=64,kernel_size=(5,1),strides=(1,1),padding='same',use_bias=True)(X)
        X = Add()([X,X_input[0]])
        return X

    OAM = {
        'name'      :'OAM',
        # Convolution Links
        'convo'    :[(0,1),(B+2,B+3),(B+4,B+5)] ,
        'convo_sub':['Conv2D','Conv2D','Conv2D',] ,
        'convo_par':[{'filters':64,'kernel_size':(9,9),'padding':repr('same'),'use_bias':True},
                     {'filters':64,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},
                     {'filters':3 ,'kernel_size':(3,3),'padding':repr('same'),'use_bias':True},],
        # Advanced Activation Links
        'aactv'    :[(1,2),],
        'aactv_sub':['PReLU',],
        'aactv_par':[{'shared_axes':(1,2)},],
        # Block Links
        'block'    :[(2+i,2+i+1)          for i in range(B)] ,
        'block_sub':['oam_residual_block' for i in range(B)] ,
        # Merge Links
        'merge'    :[(2,B+3,B+4),],
        'merge_sub':['Add'],
        }









    configs_dict = {

        'trivial_dis'          : trivial_dis ,
        'trivial_con'          : trivial_con ,

        'VDSR'                 : VDSR ,
        'vdsr_block'           : vdsr_block ,
        'vdsr_last_block'      : vdsr_last_block ,

        # SRResNet & SRGAN
        'gen_residual_block'   : gen_residual_block,
        'gen_last_block'       : gen_last_block,
        'SRResNet'             : SRResNet ,
        'SRGAN_dis'            : SRGAN_dis ,
        'baseline_con'         : baseline_con ,

        # PS_SRResNet
        'ps_gen_last_block'    : ps_gen_last_block,
        'PS_SRResNet'          : PS_SRResNet ,

        # SRResNet2
        'gen_residual_block2'  : gen_residual_block2,
        'SRResNet2'            : SRResNet2 ,

        'edsr_residual_block'  : edsr_residual_block,
        'edsr_last_block'      : edsr_last_block,
        'EDSR'                 : EDSR ,

        'oam_residual_block'   : oam_residual_block,
        'OAM'                  : OAM ,

        'drrn_residual_block'  : drrn_residual_block,
        'DRRN'                 : DRRN ,


        # E-SR-GAN ARCHITECTURE STORAGE


                    }
