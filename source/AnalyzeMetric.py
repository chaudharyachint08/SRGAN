import os
import numpy as np
import matplotlib.pyplot as plt

def eavg(ls,a=0.8):
    sum = 0
    ls2 = []
    for i in ls:
        ls2.append(a*i+(1-a)*sum)
        sum = ls2[-1]
    return np.array(ls2)

def pdiff(ls,step=5):
    ls2 = []
    for i in range(len(ls)-step):
        ls2.append( ls[i+step]-ls[i] )
    return np.array(ls2)



model_name = 'SRResNet'

st = input().strip()
if st:
    model_name = st

exp_dir = os.path.join('.','..','experiments','training_plots',model_name)

for i in os.listdir(exp_dir):
    if 'PSNR' in i and 'txt' in i:
        with open(os.path.join(exp_dir,i)) as f:
            for ln in f.readlines():
                old_set = set(globals().keys())
                exec(ln.strip())
                new_set = set(globals().keys())
                chk = (new_set-old_set).pop()
                exec('{0} = np.array({0})'.format(chk))


# Analysis on Training Data is of no use
# plt.plot(eavg(pdiff( val_PSNR),1)-eavg(pdiff(val_IPSNR),1))


def f(name=model_name,a=1,s_ix=None,e_ix=None,step=5):
    y1 = eavg(pdiff(     PSNR[s_ix:e_ix],step),a)
    y2 = eavg(pdiff(val_PSNR[s_ix:e_ix],step),a)
    # y1 = eavg(val_PSNR[s_ix:e_ix], a)
    # y2 = eavg(val_IPSNR[s_ix:e_ix],a)

    fig, ax = plt.subplots(1, 1, sharex=True)

    x = np.arange(len(y1))

    ax.plot(x+step, y1,label='Δ   PSNR' )
    ax.plot(x+step, y2,label='Δ I-PSNR')

    xmin, xmax = plt.xlim()
    ax.hlines( 0, xmin, xmax )


    xposition = np.arange(0,len(x),step)
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')

    ax.fill_between(x+step, y1, y2, where=y2 >= y1, facecolor='red', interpolate=True)
    ax.fill_between(x+step, y1, y2, where=y2 <= y1, facecolor='blue'  , interpolate=True)
    ax.set_title('fill between where')
    # plt.grid(True)

    plt.ylabel('Change Values') ; plt.title('PSNR vs IPSNR change values')
    plt.legend( loc='upper left' , bbox_to_anchor=(1,1) , fancybox=True , shadow=True )
    # plt.savefig( os.path.join(plot_dir,'{}.PNG'.format(key)) , dpi=600 , bbox_inches='tight' , format='PNG' )


    plt.show()