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
    
    baseline_gen = {
        'name'      : 'baseline_gen',
        'convo':[(0,1),]
        'convo_sub':['Conv2D']
        'convo_par':[{'filters':64, kernel_size, strides=(1, 1)}]

        }
    baseline_dis = {
        'name'      : 'baseline_dis',
        }
    baseline_con = {
        'name'      : 'baseline_con',
        }


    def conv_block(X_input):
        res = Conv2D(64,(5,5),padding='same',activation='relu')(X_input[0])
        return res

    config01 = { # 225409 Number of learnable parameters
        'name' : 'baseline_with_block',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),],
        'l_conv_type'      : ['Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : ['same','same','same','same','same','same','same',],
        'l_dilation_rate'  : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_activation'     : ['relu','relu','relu','relu','relu','relu','linear',],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),],
        'l_merge_type'   : ['Concatenate','Add',],
        # Blocks be defined here
        'l_block_connection' : [(2,3),],
        'l_block_fun' : ['conv_block',],
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


    def rec_attention_block(X_input):
        X_first = Conv2D(64,(3,3),activation='relu',padding='same')(X_input[0])
        batchnorm1, batchnorm2   = BatchNormalization(), BatchNormalization()
        activation1, activation2 = Activation('relu'),   Activation('relu')
        conv1                    = Conv2D(64,(3,3),activation='linear', padding='same')
        conv2                    = Conv2D(64,(3,3),activation=attention,padding='same')
        X = X_first
        for i in range(U):
            X1 = conv1(activation1(batchnorm1(X)))
            X2 = conv2(activation2(batchnorm2(X)))
            X = Multiply()([X1,X2])
            X = Add()([X,X_first])
        X_last = X
        return X_last

    darrn = {
        'name' : 'darrn',
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
        'l_block_fun' : ['rec_attention_block' for i in range(B)],
        }


    configs_dict = {
        'baseline_gen' : baseline_gen ,
        'baseline_dis' : baseline_dis ,
        'baseline_con' : baseline_con ,

        # 'SRCNN_L8':SRCNN_L8,
        # 'SRCNN_L4':SRCNN_L4,

        'conv_block':conv_block,
        'baseline_with_block':config01,

        'rec_block':rec_block,
        'drrn':drrn,

        'rec_attention_block':rec_attention_block,
        'darrn':darrn,        
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
'''