import os,pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


# chosen_key = 'initial_psnr'
chosen_key = 'psnr_gain'

x_axis     = 'PSNR Gain'

base_path  = '/home/malhar/my_work/Data_Analytics_Project/SRGAN/experiments/training_plots/DIV2K-Ranking'

dir_ls = os.listdir(base_path)

key1, key2 = 'Uniform Sampling', 'Our Sampling Algorithm'


def eavg(ls,a=0.1):
    ls2 = [ls[0]]
    for i in ls[1:]:
        ls2.append( a*i + (1-a)*ls2[-1] )
    return np.array(ls2)

# Window size is 10% of maximum allowed epochs which is 250 (out of 2500), eps = 0.001
def thres_ix(ls,eps=0.001,win=250):
    i, j  = 0, 0
    while j < len(ls):
        if (ls[j]-ls[i]) > 0:
            i = j
        if (j-i) >= win:
            break
        if abs(ls[i]-ls[j]) > 1:
            del ls[j]
            continue
        j+=1

    return i

data = {}
ymin, ymax = np.inf, -1*np.inf

for fldr in dir_ls:
    data[fldr] = {}
    with open(os.path.join(base_path,fldr,'IPSNR.txt')) as file:
        for line in file.readlines():
            exec(line)
        data[fldr]['IPSNR']       = IPSNR
        data[fldr]['val_IPSNR']   = val_IPSNR
        # data[fldr]['val_IPSNR'] = eavg(data[fldr]['val_IPSNR'])
        data[fldr]['th'] = thres_ix( data[fldr]['val_IPSNR'] )
        ymin = min( (ymin,min(data[fldr]['IPSNR']),min(data[fldr]['val_IPSNR'])) )
        ymax = max( (ymax,max(data[fldr]['IPSNR']),max(data[fldr]['val_IPSNR'])) )


def myplot(init_steps = 50):
    for key in data:
        plt.close()
        plt.plot( data[key][    'IPSNR'][init_steps:data[key]['th']],label='train' )
        plt.plot( data[key]['val_IPSNR'][init_steps:data[key]['th']],label='valid' )
        # plt.ylim(bottom=ymin, top=ymax)
        plt.ylim(top=ymax)
        plt.grid(True)
        plt.title(key)
        plt.xlabel('Epochs')
        plt.ylabel('PSNR Gain')
        plt.savefig( 'Sampling {}.PNG'.format(key) , dpi=600 , bbox_inches='tight' , format='PNG' )
        plt.show()
