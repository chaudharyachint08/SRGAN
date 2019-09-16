import os,pandas as pd, numpy as np

ipdir = input('Enter Directory where results are placed\n')
print( '{:10s} {:8s} {:8s} {:8s} {:8s} {:8s}'.format('DataSet Name','Minimum','Maximum','Mean','Median','Standard Deviation') )
for i in os.listdir(ipdir):
    if i.endswith('.csv'):
        df = pd.read_csv(os.path.join(ipdir,i))
        arr = df['psnr_gain']
        print( '{:10s} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(i.split()[0] , arr.min() , arr.max() , arr.mean() , arr.median() , arr.std()) )