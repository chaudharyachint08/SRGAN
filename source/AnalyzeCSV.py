import os,pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


# chosen_key = 'initial_psnr'
chosen_key = 'psnr_gain'

x_axis     = 'PSNR Gain'

base_path  = '/home/malhar/my_work/Data_Analytics_Project/SRGAN/experiments/vary_tests'


train_ls = ['DIV2K','Flickr2K','CLIC_professional','CLIC_mobile']

plot_choice = {
    'CLIC_mobile'       : ['train','valid',],
    'CLIC_professional' : ['train','valid',],
    'DIV2K'             : ['train','valid',],
    'Flickr2K'          : ['train',],

    # 'CLIC_mobile'       : ['valid',],
    # 'CLIC_professional' : ['valid',],
    # 'DIV2K'             : ['valid',],

    'BSDS100'           : ['test',],
    'Manga109'          : ['test',],
    'PIRM'              : ['valid','test',],
    'Set5'              : ['test',],
    'Set14'             : ['test',],
    'Urban100'          : ['test',],
    # '':[],
}

plot_list = []
for key in plot_choice:
    for val in plot_choice[key]:
        plot_list.append( ' '.join((key,val)) )


def ihist(data):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.5)
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=2)




def multi_train(data_name='PIRM',data_type='test'):
    print( '{:70s} {:8s} {:8s} {:8s} {:8s} {:8s}'.format('DataSet Name','Minimum','Maximum','Mean','Median','Standard Deviation') )
    for train_name in train_ls:
        ipdir = os.path.join(base_path,train_name)
        for i in sorted(os.listdir(ipdir)):
            if i.endswith('.csv') and ' '.join((data_name,data_type)) in i:
                name = ' '.join(i.split()[:2])
                if name in plot_list:
                    df = pd.read_csv(os.path.join(ipdir,i))
                    sns.distplot(df[chosen_key],bins=10,hist=False,kde=True,rug=False,label=train_name,axlabel=False)
                    arr = df['psnr_gain']
                    print( '{:70s} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(i , arr.min() , arr.max() , arr.mean() , arr.median() , arr.std()) )

    plt.grid(True)
    plt.title(' '.join((data_name,data_type)))
    plt.legend( fancybox=True , shadow=True )
    # plt.legend( loc='upper left' , bbox_to_anchor=(1,1) , fancybox=True , shadow=True )

    axs = plt.axes()
    axs.set_xlabel(x_axis)
    plt.savefig( os.path.join('Train {}.PNG'.format(' '.join((data_name,data_type)))) , dpi=600 , bbox_inches='tight' , format='PNG' )
    plt.close()
    # axs.get_yaxis().set_visible(False)
    # plt.show()


def multi_test(train_name='DIV2K'):
    ip_dir = os.path.join(base_path,train_name)

    print( '{:70s} {:8s} {:8s} {:8s} {:8s} {:8s}'.format('DataSet Name','Minimum','Maximum','Mean','Median','Standard Deviation') )
    for i in sorted(os.listdir(ipdir)):
        if i.endswith('.csv'): # and 'train' in i:
            name = ' '.join(i.split()[:2])
            if name in plot_list:
                df = pd.read_csv(os.path.join(ipdir,i))

                sns.distplot(df[chosen_key],bins=10,hist=False,kde=True,rug=False,label=name,axlabel=False)

                arr = df['psnr_gain']
                print( '{:70s} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(i , arr.min() , arr.max() , arr.mean() , arr.median() , arr.std()) )

    plt.grid(True)
    plt.legend( fancybox=True , shadow=True )
    # plt.legend( loc='upper left' , bbox_to_anchor=(1,1) , fancybox=True , shadow=True )

    axs = plt.axes()
    axs.set_xlabel(x_axis)
    plt.savefig( os.path.join('Test {}.PNG'.format(chosen_key)) , dpi=600 , bbox_inches='tight' , format='PNG' )
    # axs.get_yaxis().set_visible(False)
    plt.show()
