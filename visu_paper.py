# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:28:33 2017

@author: mducoffe

visu curve
"""

import numpy as np
import pylab as pl

#%%
# step 1 read csv file
from contextlib import closing
import csv
import os


filename="random.csv"
def get_actif_data(repository, filename, max_value=800):

    x_labels=[]
    y_acc=[]
    y_max = 0
    with closing(open(os.path.join(repository, filename))) as f:
        csv_f = csv.reader(f, delimiter=';', quotechar='|')
    
        for row in csv_f:
            x, y = int(row[0]), float(row[1])
            if x >=max_value:
                continue
            """
            if y <0.25 and x >600:
                continue
            """
            
            
            if y < y_max:
                y = y_max
            else:
                y_max = y
            
            x_labels.append(x)
            y_acc.append(y)
    return x_labels, y_acc
#%%
repository="data/csv"
dataset='BagShoes'
network='LeNet5'
repository = os.path.join(repository, '{}/{}'.format(dataset, network))
methods = ['random','aaq', 'saaq', 'uncertainty', 'bald', 'egl']
filenames =['{}_{}_'.format(dataset, network)+str(method)+'.csv' for method in methods]
#filenames=['CIFAR_VGG_random.csv', 'CIFAR_VGG_egl.csv', 'CIFAR_LeNet5_uncertainty.csv']
legends=methods
linestyles=['r-', 'b--', 'b-', 'g-', 'k-', 'c-', 'k-']
dico_actif={}

for filename, legend, linestyle in zip(filenames, legends, linestyles):
    actif_key=filename.split('.csv')[0]
    print((actif_key, linestyle))
    dico_actif[actif_key]=[get_actif_data(repository, filename), legend, linestyle]

#%%
pl.figure(1)
pl.clf()
for key in dico_actif:
    data, legend, linestyle = dico_actif[key]
    x_labels, y_acc = data
    pl.plot(x_labels,y_acc,linestyle, label=legend)
    pl.hold(True)
    
    
pl.grid()
pl.hold(False)


pl.legend(bbox_to_anchor=(0.5, 0.6), loc=2, borderaxespad=0.)
pl.savefig('img/test_acc_{}_{}.pdf'.format(dataset, network), dpi=300, bbox_inches='tight')
#pl.plot(ytest,yest,'+')
#%%


