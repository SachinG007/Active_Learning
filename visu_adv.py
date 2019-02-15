# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:01:48 2017

@author: mducoffe

# visu adv
"""
import pickle as pkl
from contextlib import closing

#%%

def load_adv(repo, filename):
    i = 0
    assert os.path.isdir(repo), ('unknown repository %s', repo)
    while os.path.isfile(os.path.join(repo, filename+'_'+str(i)+'.pkl')):
        i+=1
        
    filenames = [os.path.join(repo, filename+'_'+str(j)+'.pkl') for j in range(1,i)]
    img_real=[]; img_adv=[]
    for filename in filenames:
        print(filename)
        with closing(open(os.path.join(repo, filename), 'rb')) as f:
            img_0, img_1 =pkl.load(f)
            img_real.append(img_0)
            img_adv.append(img_1)
    return img_real, img_adv

#%%
repository="."
dataset='MNIST'
network='LeNet5'
active='aaq'

filename_template = 'adv_{}_{}_{}'.format(dataset, network, active)
img_real, img_adv =load_adv(repo, filename_template)
#%%

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.   # the bottom of the subplots of the figure
top = 0.5  # the top of the subplots of the figure
wspace = 0.01   # the amount of width reserved for blank space between subplots
hspace = 0.




toto = img_real[0]
tata = img_adv[0]

img_real = [toto]*5
img_adv = [tata+10]*5
perturbations = [ real - adv for (real, adv) in zip(img_real, img_adv)]
import pylab as pl

#pl.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
def hide_axis():
    """hides axis but let you use xlabel and ylalbels"""
    pl.gca().spines['bottom'].set_color('white')
    pl.gca().spines['top'].set_color('white') 
    pl.gca().spines['right'].set_color('white')
    pl.gca().spines['left'].set_color('white')
    pl.xticks(())
    pl.yticks(())

nb_data=10
nb_query=10  
nb_rows=3
nb_cols=N
N=5
pl.figure(1, (nb_rows, nb_cols))
for i in range(N):
    pl.subplot(nb_rows,nb_cols,i+1)
    pl.imshow(img_real[i][0], cmap='Blues',interpolation='nearest')
    if i==0:
        pl.ylabel('Top Score \n Query',fontsize=9)
    hide_axis()

for i in range(N):
    pl.subplot(nb_rows,nb_cols,i+1+N)
    pl.imshow(img_adv[i][0], cmap='Blues',interpolation='nearest')
    if i==0:
        pl.ylabel('Adv \n Attack',fontsize=9)
    hide_axis()

for i in range(N):
    pl.subplot(nb_rows,nb_cols,i+1+2*N)
    pl.imshow(perturbations[i][0], cmap='Blues',interpolation='nearest')
    if i==0:
        pl.ylabel('Adv \n Noise',fontsize=9)
    pl.xlabel(str(nb_data+(i+1)*nb_query), fontsize=9)
    hide_axis()
pl.tight_layout(pad=0,h_pad=-20,w_pad=0) 



pl.close(5)
pl.figure(5,(nbt,nbm))
pl.clf()

for m in range(nbm):
    
    for i in range(nbt):
        pl.subplot(nbm,nbt,i+1+m*nbt)
        pl.imshow(-Bi[m][i,:,:],cmap='gray')
        
        if i==0:
            pl.ylabel(methods[m],fontsize=18)
        
        hide_axis()
pl.tight_layout(pad=0,h_pad=-.5,w_pad=-1.5)     
pl.savefig('imgs/interp_comp_{}.pdf'.format(expe),dpi=300,bbox_inches='tight')  



