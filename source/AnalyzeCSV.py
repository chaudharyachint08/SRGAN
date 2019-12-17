import os,pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


# chosen_key = 'initial_psnr'
# x_axis     = 'PSNR (BICUBIC)'


chosen_key = 'psnr_gain'
x_axis     = 'PSNR Gain'

base_path  = '../experiments(analytics)/vary_tests'


train_ls = ['CLIC_mobile','CLIC_professional','DIV2K','Flickr2K',]

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
    'Multiple trained model instances performance on same test set'
    print( '{:70s} {:8s} {:8s} {:8s} {:8s} {:8s}'.format('DataSet Name','Minimum','Maximum','Mean','Median','Standard Deviation') )
    for i in os.listdir(os.path.join(base_path,'DIV2K')):
        for train_name in sorted(train_ls):
            if i.endswith('.csv') and ' '.join((data_name,data_type)) in i:
                name = ' '.join(i.split()[:2])
                if name in plot_list:
                    df = pd.read_csv(os.path.join(base_path,train_name,i))
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
    'Same model performance is plotted for all test sets in dcitionary described above'
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
