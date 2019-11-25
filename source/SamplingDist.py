import os,pandas as pd, numpy as np
from scipy.stats import norm
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

import seaborn as sns


plot_dir = 'GIF'
if plot_dir in os.listdir():
    shutil.rmtree(plot_dir)
os.mkdir(plot_dir)

epochs = 2000
k,N = 10, 1000

min_psnr, max_psnr = 20,40
bicubic_bins, prob_update_bins = 20, 5

prob_upper_limit = 0.003
# init_option = 'Uniform'
init_option = 'Bicubic'



psnr = np.random.randn(N)
psnr = min_psnr + ((psnr-psnr.min())/(psnr.max()-psnr.min()))*(max_psnr-min_psnr)




def uniform_prob(psnr):
    'Assigns Uniform Probability to Each Patch'
    return np.ones(len(psnr)) / len(psnr)

def bicubic_prob(psnr,bins=bicubic_bins):
    'Assigns Probability using Initial PSNR using BICUBIC to Each Patch'
    freq, ranges, _ = plt.hist(psnr,bins=bins)
    plt.close()
    prob = []
    for psnr_val in psnr:
        L,U = 0, len(freq)
        while L <= U:
            mid = (L+U)//2
            if ranges[mid] < psnr_val and psnr_val <= ranges[mid+1]:
                break
            elif psnr_val <= ranges[mid]:
                U = mid - 1
            else:
                L = mid + 1
        prob.append( sum(  freq[i]*(ranges[i+1]-ranges[i])  for i in range(mid,len(freq)) ) )
    prob = np.array(prob)
    return prob / prob.sum()

def sample(prob_vector,size=k):
    'Return ix list of size, given Probability vector'
    prob_vector = prob_vector / prob_vector.sum()
    cprob_vector = np.cumsum(prob_vector)
    ix_ls = []
    for _ in range(k):
        val = np.random.random()
        L, U = 0,len(cprob_vector)-1
        while True:
            mid = (L+U)//2
            if (val <= cprob_vector[mid]) and (mid==0 or (cprob_vector[mid-1]<val)):
                break
            elif (val <= cprob_vector[mid]):
                U = mid-1
            else:
                L = mid+1
            # print(val)
        if mid > len(cprob_vector)-1:
            mid = len(cprob_vector)-1
        ix_ls.append( mid )
    return ix_ls

def update_prob(psnr,prob_vector,sample_ix_ls):
    'Update probability for sample using Bicubic like apporach'
    sample_psnr, sample_old_prob = psnr[sample_ix_ls], prob_vector[sample_ix_ls]
    sample_new_prob = bicubic_prob(sample_psnr,bins=prob_update_bins)*sample_old_prob.sum()
    prob_vector[sample_ix_ls] = sample_new_prob

def psnr_prob_hist(psnr,prob_vector):
    'Function to find mean value of Probability in each bin of PSNR'
    # print(prob_vector.min(),prob_vector.max())
    # print(psnr.min(),psnr.max())

    all_ix = np.arange(len(psnr))
    X = [(psnr_ranges[i]+psnr_ranges[i+1])/2  for i in range(len(psnr_ranges)-1)]

    Y = []
    for i in range(len(psnr_ranges)-1):
        low, high = psnr_ranges[i], psnr_ranges[i+1]
        if i==0:
            bool_range = 1*( (psnr <= high) )
            Y.append( (prob_vector*bool_range).sum() / bool_range.sum() )
        elif i == (len(psnr_ranges)-1):
            bool_range = 1*( (low < psnr) )
            Y.append( (prob_vector*bool_range).sum() / bool_range.sum() )
        else:
            bool_range = 1*( (low < psnr) & (psnr <= high) )
            Y.append( (prob_vector*bool_range).sum() / bool_range.sum() )
    return np.array(X), np.array(Y)





psnr_freq, psnr_ranges, _ = plt.hist(psnr,bins=bicubic_bins,color='green')
plt.grid(True)
plt.xlabel('PSNR (BICUBIC)')
plt.ylabel('Number of Occurence')
plt.title('Data Histogram')
plt.savefig( 'Data Histogram.PNG' , dpi=600 , bbox_inches='tight' , fromat='PNG' )
# plt.show()
plt.close()


if init_option == 'Bicubic':
    prob_vector = bicubic_prob(psnr)
else:
    prob_vector = uniform_prob(psnr)


X, Y = psnr_prob_hist(psnr,prob_vector)
plt.bar( x=X,height=Y,width=(psnr_ranges[1]-psnr_ranges[0]),align='center', color='orange' )
plt.grid(True)
plt.xlabel('PSNR (BICUBIC)')
plt.ylabel('Probability')
plt.title('Assignment using {} Distribution'.format(init_option))
plt.savefig( '{} Assignment.PNG'.format(init_option) , dpi=600 , bbox_inches='tight' , fromat='PNG' )
# plt.show()
plt.close()




for _ in range(epochs):
    plt.close()
    fig, ax = plt.subplots()
    ax.set_ylim(0, prob_upper_limit)
    X, Y = psnr_prob_hist(psnr,prob_vector)

    plt.bar(X,Y, width=(psnr_ranges[1]-psnr_ranges[0]), align='center')#, color='skyblue')
    plt.grid(True)
    plt.xlabel('PSNR (BICUBIC)')
    plt.ylabel('Probability')
    plt.title('Epoch Number {}'.format(_+1))

    plt.savefig( os.path.join(plot_dir,'{}.PNG'.format(_)) , dpi=400 , bbox_inches='tight' , format='PNG' )
    sample_ix_ls = sample(prob_vector)
    # print(sorted(psnr[sample_ix_ls]))
    update_prob(psnr,prob_vector,sample_ix_ls)

