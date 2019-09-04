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
    
    config0 = { # 225409 Number of learnable parameters
        'name' : 'baseline',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),],
        'l_conv_type'      : ['Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [64,64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [(5,5),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : ['same','same','same','same','same','same','same','same',],
        'l_dilation_rate'  : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_activation'     : ['relu','relu','relu','relu','relu','relu','relu','linear',],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),],
        'l_merge_type'   : ['Concatenate','Add',],
        }

    ######## NON-WORKING OR SAME COMPLEXITY MODELS BEGINS ########
    config1 = { # 225921 Number of learnable parameters, even some more complexity
        'name' : 'custom_model_1',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [(2,3),(3,4),(4,5),(6,7),(7,8),(9,10),(11,12)],
        'l_conv_type'      : ['Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [(5,5),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : ['same','same','same','same','same','same','same',],
        'l_dilation_rate'  : [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_activation'     : ['relu','relu','relu','relu','relu','relu','linear',],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,12,13),(3,5,6),(4,8,9),(7,10,11)],
        'l_merge_type'   : ['Concatenate','Add','Add','Concatenate','Concatenate'],
        }
    config5 = {
        'name' : 'custom_model_5',
        # Attributes of Each Convolutional layer be defined here, non-feasible for heterogenous sized images
        'l_conv_connection': [ (2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(10,11)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,len(channel_indx),len(channel_indx),],
        'l_kernel_size'    : [ (5,5),(3,1),(1,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (2,2),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'PReLU','relu','relu','relu','PReLU','PReLU','PReLU'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,9,10),(3,6,7),],
        'l_merge_type'   : ['Concatenate','Concatenate','Concatenate',],
        }
    ######## NON-WORKING OR SAME COMPLEXITY MODELS BEGINS ########



    config2 = { # 144010 Number of learnable parameters, 1.565x Reduction
        'name' : 'custom_model_2',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(10,11)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [ (5,5),(3,1),(1,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,9,10),(3,6,7),],
        'l_merge_type'   : ['Concatenate','Concatenate','Concatenate',],
        }

    config4 = { # 106762 Number of learnable parameters, 2.09x Reduction
        'name' : 'custom_model_4',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(10,11)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,32,32,32,72,48,len(channel_indx),],
        'l_kernel_size'    : [ (5,5),(3,1),(1,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,9,10),(3,6,7),],
        'l_merge_type'   : ['Concatenate','Concatenate','Concatenate',],
        }



    config3 = { # 106762 Number of learnable parameters, 1.64x Reduction
        'name' : 'custom_model_3',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(9,10)],
        'l_conv_type'      : [ 'SeparableConv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [ (3,3),(3,1),(1,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (2,2),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'linear','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(2,6,7),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }

    
    config6 = { # 106762 Number of learnable parameters, 1.38x Reduction
        'name' : 'custom_model_6',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(2,6),(6,7),(8,9),(9,10)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,64,len(channel_indx),],
        'l_kernel_size'    : [ (3,1),(1,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(5,7,8),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }


    config7 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'custom_model_7',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(2,6),(6,7),(8,9),(9,10)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','SeparableConv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,72,len(channel_indx),],
        'l_kernel_size'    : [ (3,1),(1,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(2,2),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(5,7,8),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }


    config8 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'custom_model_8',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(2,6),(6,7),(8,9),(9,10)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,48,64,48,72,len(channel_indx),],
        'l_kernel_size'    : [ (3,1),(1,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(2,2),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(5,7,8),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }


    config9 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'custom_model_9',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(2,6),(6,7),(8,9),(9,10)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','SeparableConv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,64,64,64,72,len(channel_indx),],
        'l_kernel_size'    : [ (3,1),(1,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(2,2),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'relu','relu','relu','relu','relu','relu','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(5,7,8),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }


    config10 = { # 106762 Number of learnable parameters, 2.1x Reduction
        'name' : 'custom_model_10',
        # Attributes of Each Convolutional layer be defined here
        'l_conv_connection': [ (2,3),(3,4),(4,5),(2,6),(6,7),(8,9),(9,10)],
        'l_conv_type'      : [ 'Conv2D','Conv2D','SeparableConv2D','Conv2D','Conv2D','Conv2D','Conv2D',],
        'l_filters'        : [ 64,64,48,64,48,72,len(channel_indx),],
        'l_kernel_size'    : [ (3,1),(1,3),(3,3),(3,3),(3,3),(3,3),(3,3),],
        'l_strides'        : [ (1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),],
        'l_padding'        : [ 'same','same','same','same','same','same','same'],
        'l_dilation_rate'  : [ (1,1),(1,1),(1,1),(2,2),(1,1),(1,1),(1,1)],
        'l_activation'     : [ 'LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','LeakyReLU','linear'],
        # Connections be defined here
        'l_merge_connection' : [(0,1,2),(0,10,11),(5,7,8),], # Concatenating 2nd instead of any other makes less parameters
        'l_merge_type'   : ['Concatenate','Add','Concatenate',],
        }


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

    def conv_block(X_input):
        res = Conv2D(64,(5,5),padding='same',activation='relu')(X_input)
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
        X_first = Conv2D(64,(3,3),activation='relu',padding='same')(X_input)
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
        X_first = Conv2D(64,(3,3),activation='relu',padding='same')(X_input)
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
        'baseline':config0,

        'custom_model_1':config1, # Almost same complexity
        'custom_model_5':config5, # Test Time unfeasible if image sizes varies, also PReLU adds many parameters

        'custom_model_2':config2,
        'custom_model_4':config4,

        'custom_model_3':config3,
        'custom_model_6':config6,

        'custom_model_7':config7,
        'custom_model_8':config8,

        'custom_model_9':config9, # Model 7, but for AMIX
        'custom_model_10':config10, # Model 8, last linear, for AMIX

        'SRCNN_L8':SRCNN_L8,
        'SRCNN_L4':SRCNN_L4,

        'conv_block':conv_block,
        'baseline_with_block':config01,

        'rec_block':rec_block,
        'drrn':drrn,

        'rec_attention_block':rec_attention_block,
        'darrn':darrn,        
                    }
