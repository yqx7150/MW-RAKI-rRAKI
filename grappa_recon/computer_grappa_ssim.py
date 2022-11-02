#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import scipy.io as sio
from  skimage.metrics import peak_signal_noise_ratio
from  skimage.metrics import structural_similarity
from  skimage.metrics import mean_squared_error
from math import sqrt





result=sio.loadmat('/media/lqg/KESU/TH/kspace_learninh_生成权重/grappa_recon/grappa_recons/kspace_full.mat')
recon=sio.loadmat('/media/lqg/KESU/TH/kspace_learninh_生成权重/grappa_recon/grappa_recons/kspace_grappa.mat')



truth=result['kspace']
# truth=np.transpose(truth,(1,0))
all_raki=recon['kspace_grappa']
# all_raki=np.transpose(all_raki,(1,0))
print(truth.shape)
print(all_raki.shape)


# In[11]:


def rsos(x,axis):
    '''
    Computes root sum of square along the specified axis
    '''
    return np.sqrt(np.sum(np.square(np.abs(x)),axis = axis))
def rmse(original,comparison):
    '''
    Computes the normalized root mean squared error between an original (ground truth) object and a comparison.  As long as the object are of the same dimension, this function will vectorize and compute the desired value
    '''
    return np.sqrt(np.sum(np.square(np.abs(original-comparison))))/np.sqrt(np.sum(np.square(np.abs(original))))


truth=rsos(truth,axis=-1)

all_raki=rsos(all_raki,axis=-1)



##单独测一个的时候打开
import matplotlib.pyplot as plt
# print(np.max(truth/np.max(truth)))
# print(np.max(all_raki/np.max(all_raki)))
truth=truth/np.max(truth)
all_raki=all_raki/np.max(all_raki)
plt.subplot(121)
plt.imshow(truth,cmap='gray',vmin=0,vmax=1)

plt.subplot(122)
plt.imshow(all_raki,cmap='gray',vmin=0,vmax=1)
plt.show()

#单独一个的
psnr=peak_signal_noise_ratio(truth,all_raki)
ssim=structural_similarity(truth,all_raki)
rmse = rmse(truth, all_raki)*100

print('psnr',psnr)
print('ssim',ssim)
print('rmse',rmse)


# In[154]:


# for i in range(6):
#     erro_1=truth-all_raki[:,:,i]
#     plt.figure(i)
#     plt.imshow(erro_1*10,cmap='gray',vmin=0,vmax=1)
#     plt.show()
#     print(np.max(erro_1))

# erro_1=truth-all_raki
# plt.figure(11)
# plt.imshow(erro_1*10,cmap='gray',vmin=0,vmax=1)
# plt.show()
# print(np.max(erro_1))





