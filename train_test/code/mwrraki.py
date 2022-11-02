#!/usr/bin/env python
# coding: utf-8

# Perform residual RAKI reconstruction on a variety of ACS sizes and accelerations (similar for what I did in SPARK and RAKI, for sake of experimental comparison)

# In[1]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import numpy as np
import numpy.matlib
import time
import os
from  skimage.metrics import peak_signal_noise_ratio,structural_similarity
from utils import signalprocessing as sig
import matplotlib.pyplot as plt

# ### RAKI Definitions 

# In[2]:


def weight_variable(shape,vari_name):                   
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d_same(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_dilate_same(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='SAME',dilation_rate = [1,dilate_rate])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

def learning_residual_raki(ACS_input,target_input,accrate_input,sess,    ACS_dim_X,ACS_dim_Y,ACS_dim_Z,target_dim_X,target_dim_Y,target_dim_Z,    target,target_x_start,target_x_end,target_y_start,target_y_end,    ACS):
    input_ACS    = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                  
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z]) 
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)
    
    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input))

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    # h_conv3 = conv2d_dilate(h_conv1, W_conv3,accrate_input)
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)

    W_conv_linear = weight_variable([kernel_x_linear,kernel_y_linear,ACS_dim_Z,target_dim3],'W_lin')
    h_linear = conv2d_dilate(Input,W_conv_linear,accrate_input)
    x_length = h_conv3.shape[1]
    y_length = h_conv3.shape[2]
    
    if(y_length % 2 == 0):
        h_linear = h_linear[:,x_length//2 - x_length//2:x_length//2 + x_length//2,            y_length//2 - y_length//2:y_length//2 + y_length//2,:]
    else:
        h_linear = h_linear[:,x_length//2 - x_length//2:x_length//2 + x_length//2,            y_length//2 - y_length//2:y_length//2 + y_length//2+1,:]
        
    # error_norm = tf.norm(input_Target - h_linear) + tf.norm(input_Target - h_linear - h_conv3)
    error_norm = tf.norm(input_Target - h_linear) + tf.norm(input_Target - h_linear - h_conv3)
    train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)
    
     
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1 
    for i in range(MaxIteration):
        
        sess.run(train_step, feed_dict={input_ACS: ACS_input, input_Target: target_input})
        if i % 100 == 0:                                                                      
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS_input, input_Target: target_input})
            print('The',i,'th iteration gives an error',error_now)  
            
    error = sess.run(error_norm,feed_dict={input_ACS: ACS_input, input_Target: target_input})
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),sess.run(W_conv_linear),error]
    
def cnn_linear(input_kspace,w_linear,acc_rate,sess):
    return sess.run(conv2d_dilate(input_kspace,w_linear,acc_rate))

def cnn_3layer(input_kspace,w1,w2,w3,acc_rate,sess):
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate)
    return sess.run(h_conv3)           


# ### Loading k-space and defining fft operators

# In[3]:


fft2c_raki  = lambda x: sig.fft(sig.fft(x,0),1)
ifft2c_raki = lambda x: sig.ifft(sig.ifft(x,0),1)

# image_coils_truth  = sio.loadmat('data/img_grappa_32chan.mat')['IMG']
image_coils_truth  = sio.loadmat('data/256DATA1/3.mat')['Img']
# image_coils_truth  = sio.loadmat('data/216DATA/111.mat')['Img']
# image_coils_truth  = sio.loadmat('data/2DMREPPA2.mat')['Img']
##################################################################
# normalize = 1/np.max(abs(image_coils_truth[:]))
# image_coils_truth = np.multiply(image_coils_truth, normalize)
normalize = 1/np.max(abs(image_coils_truth[:]))
image_coils_truth = np.multiply(image_coils_truth, normalize)

kspace_truth_raki  = fft2c_raki(image_coils_truth)
novcc_kspace_truth_raki=np.copy(kspace_truth_raki)


[M,N,C] = kspace_truth_raki.shape

weight=np.repeat(sio.loadmat('data/256weight/weight7.mat')['weight'][:,:,np.newaxis],C,axis=-1)
weight1=np.repeat(sio.loadmat('data/256weight/weight7.mat')['weight'][:,:,np.newaxis],C,axis=-1)

# ### Setting Non-changing Experimental Parameters 

# In[4]:


GPU_FRAC = 1/4

Rx     = 1
acsx   = M

#### Linear Network Parameters ####
kernel_x_linear = 5
kernel_y_linear = 2

#### RAKI Network Parameters ####
kernel_x_1 = 5
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = C
layer2_channels = 8

MaxIteration = 250
LearningRate = 1e-3


# ### Setting ablation parameters

# In[5]:


all_Ry    = [6]
all_acsy  = [24]
             
all_parameters = []

for ry in all_Ry:
    for acsy in all_acsy:
        all_parameters.append({'Ry':ry, 'acsy':acsy})


# ### Defining residual-RAKI reconstruction function (which will be looped over the parameters)

# In[6]:


def residual_raki(MaxIteration,LearningRate,Rx,Ry,acsx,acsy,GPU_FRAC,    kspace_truth_raki=kspace_truth_raki,kernel_x_linear=kernel_x_linear,    kernel_y_linear=kernel_y_linear,kernel_x_1=kernel_x_1,kernel_y_1=kernel_y_1,kernel_x_2=kernel_x_2,    kernel_y_2=kernel_y_2,kernel_last_x=kernel_last_x,kernel_last_y=kernel_last_y,
    layer1_channels=layer1_channels,layer2_channels=layer2_channels):
    
    acsregionX = np.arange(M//2 - acsx // 2,M//2 + acsx//2)
    global acsregionY
    acsregionY = np.arange(N//2 - acsy // 2,N//2 + acsy//2) 

    kspace_raki_undersampled_withacs = np.zeros((M,N,C),dtype = complex)
    kspace_raki_undersampled_withacs[::Rx,::Ry,:] = kspace_truth_raki[::Rx,::Ry,:]
    kspace_raki_undersampled_withacs[acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,:]        = kspace_truth_raki[acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,:]

    kspace = np.copy(kspace_raki_undersampled_withacs)

    kspace_no_nor = np.copy(kspace)

    no_ACS_flag = 0;

    # normalize = 0.015/np.max(abs(kspace[:]))
    # kspace = np.multiply(kspace,normalize)

    kspace_0_w = np.copy(kspace_no_nor)
    kspace_1_w = np.copy(kspace_no_nor)

    kspace_0_w_a = np.multiply(kspace_0_w, weight);kspace_0_w_a = np.copy(kspace_0_w_a)
    kspace_1_w_a = np.multiply(kspace_1_w, weight1); kspace_1_w_a = np.copy(kspace_1_w_a)

    [m1,n1,no_ch] = np.shape(kspace)
    no_inds = 1

    kspace_all = np.copy(kspace);
    kx = np.transpose(np.int32([(range(1,m1+1))]))                  
    ky = np.int32([(range(1,n1+1))])

    kspace = np.copy(kspace_all)
    mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; 
    picks = np.where(mask == 1);                                  
    kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
    kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  

    #kspace_NEVER_TOUCH = np.copy(kspace_all)
    kspace_NEVER_TOUCH = np.copy(kspace_no_nor)

    mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0;  
    picks = np.where(mask == 1);                                  
    d_picks = np.diff(picks,1)  
    indic = np.where(d_picks == 1);

    mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;
    picks_x = np.where(mask_x == 1);
    x_start = picks_x[0][0]
    x_end = picks_x[0][-1]


    no_ACS_flag=0;
    print('ACS signal found in the input data')
    indic = indic[1][:]
    center_start = picks[0][indic[0]];
    center_end = picks[0][indic[-1]+1];
    ACS_0_w = kspace_0_w_a[x_start:x_end + 1, center_start:center_end + 1, :]
    ACS_1_w = kspace_1_w_a[x_start:x_end + 1, center_start:center_end + 1, :]
    # ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    ACS=np.concatenate([ACS_0_w[np.newaxis,:,:,:],  ACS_1_w[np.newaxis,:,:,:]],axis=0)
    [ACS_batch,ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_batch,ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,:,no_ch:no_ch*2] = np.imag(ACS)

    acc_rate = d_picks[0][0]
    no_channels = ACS_dim_Z*2

    time_ALL_start = time.time()

    [ACS_batch,ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
    ACS = np.reshape(ACS_re, [ACS_batch,ACS_dim_X, ACS_dim_Y, ACS_dim_Z])
    ACS = np.float32(ACS)  
    
    w_linear_all =         np.zeros([kernel_x_linear, kernel_y_linear, no_channels, acc_rate - 1, no_channels*ACS_batch],dtype=np.float32)
        
    w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels*ACS_batch],dtype=np.float32)
    w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels*ACS_batch],dtype=np.float32)
    w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels*ACS_batch],dtype=np.float32)

    target_x_start = np.int32(np.ceil(kernel_x_1/2)  + np.floor(kernel_x_2/2)+ np.floor(kernel_last_x/2) -1);
    target_x_end = np.int32(ACS_dim_X - target_x_start -1); 

    target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1)+ (np.ceil(kernel_last_y/2)-1)) * acc_rate;
    target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2)+ np.floor(kernel_last_y/2))) * acc_rate -1;

    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_Z = acc_rate - 1
    
    print('go!')
    time_Learn_start = time.time() 

    errorSum = 0;
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC ; 
    for ii_for in range(ACS_batch):
        for ind_c in range(ACS_dim_Z):

            sess = tf.Session(config=config)
            # set target lines
            target = np.zeros([ACS_batch,target_dim_X,target_dim_Y,target_dim_Z])
            print('learning channel #',ind_c+1)
            time_channel_start = time.time()

            for ind_acc in range(acc_rate-1):
                target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1
                target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
                first_fild = ACS[0, target_x_start:target_x_end + 1, target_y_start:target_y_end + 1, ind_c];
                second_fild = ACS[1, target_x_start:target_x_end + 1, target_y_start:target_y_end + 1, ind_c];
                truth_ACS = np.concatenate([first_fild[np.newaxis, :, :], second_fild[np.newaxis, :, :]], axis=0)
                target[:, :, :, ind_acc] = truth_ACS
                #target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

            # learning
            print('第几块学习%d', ii_for)
            [w1,w2,w3,w_linear,error]=learning_residual_raki(np.reshape(ACS[ii_for,...], [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z]),np.reshape(target[ii_for,...],[1, target_dim_X,target_dim_Y,target_dim_Z]),acc_rate,sess,            ACS_dim_X,ACS_dim_Y,ACS_dim_Z,target_dim_X,target_dim_Y,target_dim_Z,target,            target_x_start,target_x_end,target_y_start,target_y_end,            ACS)
            w1_all[:,:,:,:,ind_c+ACS_dim_Z*ii_for] = w1
            w2_all[:,:,:,:,ind_c+ACS_dim_Z*ii_for] = w2
            w3_all[:,:,:,:,ind_c+ACS_dim_Z*ii_for] = w3
            w_linear_all[:,:,:,:,ind_c+ACS_dim_Z*ii_for] = w_linear

            time_channel_end = time.time()
            print('Time Cost:',time_channel_end-time_channel_start,'s')
            print('Norm of Error = ',error)
            errorSum = errorSum + error

            sess.close()
            tf.reset_default_graph()

    time_Learn_end = time.time();

    print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min')    
    kspace_recon_all = np.copy(kspace_all)
    kspace_recon_all_nocenter = np.copy(kspace_all)

    kspace = np.copy(kspace_all)

    over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
    kspace_und_0_w_a = kspace_0_w_a;kspace_und_0_w_a[:, over_samp, :] = 0;
    kspace_und_1_w_a = kspace_1_w_a;kspace_und_1_w_a[:, over_samp, :] = 0;
    kspace_und = np.copy(
        np.concatenate((kspace_und_0_w_a[np.newaxis, :, :, :], kspace_und_1_w_a[np.newaxis, :, :, :]), axis=0))

    [dim_kspaceUnd_batch,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

    kspace_und_re = np.zeros([dim_kspaceUnd_batch,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_und_re[:,:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
    kspace_und_re[:,:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
    kspace_und_re = np.float32(kspace_und_re)
    kspace_und_re = np.reshape(kspace_und_re,[dim_kspaceUnd_batch,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_recon = kspace_und_re

    kspace_recon_linear       = np.copy(kspace_recon)
    kspace_recon_residual     = np.zeros(kspace_recon.shape,dtype = complex)
        
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRAC ;

    for iii in range(ACS_batch):
        for ind_c in range(0,no_channels):
            print('Reconstruting Channel #',ind_c+1)

            sess = tf.Session(config=config)
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)

            # grab w and b
            w1 = np.float32(w1_all[:,:,:,:,ind_c+ACS_dim_Z*iii])
            w2 = np.float32(w2_all[:,:,:,:,ind_c+ACS_dim_Z*iii])
            w3 = np.float32(w3_all[:,:,:,:,ind_c+ACS_dim_Z*iii])

            w_linear = np.float32(w_linear_all[:,:,:,:,ind_c+ACS_dim_Z*iii])

            residual_recon = cnn_3layer(np.reshape(kspace_und_re[iii,...],[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2]),w1,w2,w3,acc_rate,sess)
            linear_recon   = cnn_linear(np.reshape(kspace_und_re[iii,...],[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2]),w_linear,acc_rate,sess)
            x_length = residual_recon.shape[1]
            y_length = residual_recon.shape[2]
            linear_recon = linear_recon[:,x_length//2 - x_length//2:x_length//2 + x_length//2,                y_length//2 - y_length//2:y_length//2 + y_length//2,:]
            #linear_recon = np.copy(residual_recon)

            target_x_end_kspace = dim_kspaceUnd_X - target_x_start;

            for ind_acc in range(0,acc_rate-1):
                target_y_start =                 np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) +                 np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;

                target_y_end_kspace =                 dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2))+ np.floor(kernel_last_y/2)) * acc_rate + ind_acc;

                kspace_recon[iii:iii+1,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] =                 linear_recon[0,:,::acc_rate,ind_acc] + residual_recon[0,:,::acc_rate,ind_acc];
                kspace_recon_linear[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c]                 = linear_recon[0,:,::acc_rate,ind_acc];
                kspace_recon_residual[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] =                 residual_recon[0,:,::acc_rate,ind_acc]

    kspace_recon          = np.squeeze(kspace_recon)
    kspace_recon_linear   = np.squeeze(kspace_recon_linear)
    kspace_recon_residual = np.squeeze(kspace_recon_residual)
    
    kspace_recon_complex = (kspace_recon[:,:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,:,np.int32(no_channels/2):no_channels],1j))

    kspace_recon_complex_0_w_a = kspace_recon_complex[0, :, :, :]
    kspace_recon_complex_1_w_a = kspace_recon_complex[1, :, :, :]

    kspace_recon_complex_0_w_d = np.multiply(kspace_recon_complex_0_w_a, 1. / weight)
    kspace_recon_complex_1_w_d = np.multiply(kspace_recon_complex_1_w_a, 1. / weight1)





    kspace_recon_complex_linear = (kspace_recon_linear[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon_linear[:,:,np.int32(no_channels/2):no_channels],1j))

    kspace_recon_complex_residual = (kspace_recon_residual[:,:,:,0:np.int32(no_channels/2)] +                 np.multiply(kspace_recon_residual[:,:,:,np.int32(no_channels/2):no_channels],1j))
    
    # kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex);


    if no_ACS_flag == 0:
        kspace_recon_complex_0_w_d[:, center_start:center_end, :] = kspace_0_w[:, center_start:center_end, :]
        kspace_recon_complex_1_w_d[:, center_start:center_end, :] = kspace_1_w[:, center_start:center_end, :]
        # kspace_recon_complex_2_w_d[:,center_start:center_end,:]=kspace_2_w[:,center_start:center_end,:]

        kspace_recon_complex_0_w_d[weight == 0] = kspace_0_w[weight == 0]
        kspace_recon_complex_1_w_d[weight1 == 0] = kspace_1_w[weight1 == 0]


        # kspace_recon_complex_linear[:,center_start:center_end,:] =             kspace_NEVER_TOUCH[:,center_start:center_end,:]
        # kspace_recon_complex_linear[weight == 0] = kspace_0_w[weight == 0]
        #kspace_recon_complex_residaul[:,center_start:center_end,:] = \
        #    kspace_NEVER_TOUCH[:,center_start:center_end,:]
        print('ACS signal has been putted back')
    else:
        print('No ACS signal is putted into k-space')

        sess.close()
        tf.reset_default_graph()
        
    return [kspace_recon_complex_0_w_d,kspace_recon_complex_1_w_d,kspace_recon_complex_residual]


# ### Defining loop over the reconstruction parameters

# In[ ]:


kspace_rraki_recon_all  = np.zeros((M,N,2*C,len(all_parameters)),dtype = complex)
kspace_linear_recon_all = np.zeros((M,N,C,len(all_parameters)),dtype = complex)
kspace_residual_est_all = np.zeros((M,N,C,len(all_parameters)),dtype = complex)

for index,parameter in enumerate(all_parameters):
    Ry   = parameter['Ry']
    acsy = parameter['acsy']
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Training %d/%d || Ry %d || Acsy %d' % (index+1,len(all_parameters),Ry,acsy))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    [kspace_rraki_0,kspace_rraki_1,kspace_residual] =         residual_raki(MaxIteration,LearningRate,Rx,Ry,acsx,acsy,GPU_FRAC)
    
    kspace_rraki_recon_all[:,:,:,index]     = np.concatenate([kspace_rraki_0,kspace_rraki_1],axis=-1)
    # kspace_linear_recon_all[:,:,:,index]    = kspace_linear

    kspace_residual_est_all[:,:,:,index]    = kspace_residual[1,...]


# ### Computing all reconstructions and rmse's to the corresponding ground truth 

# In[ ]:
sig.kerro(novcc_kspace_truth_raki, np.squeeze(kspace_rraki_recon_all), acsregionY, acsy)
all_recons_rraki      = np.transpose(sig.rsos(ifft2c_raki(kspace_rraki_recon_all),-2),(1,0,2))
# all_recons_linear     = np.transpose(sig.rsos(ifft2c_raki(kspace_linear_recon_all),-2),(1,0,2))
all_est_residual     = np.transpose(sig.rsos(ifft2c_raki(kspace_residual_est_all),-2),(1,0,2))

truth = np.transpose(sig.rsos(ifft2c_raki(kspace_truth_raki),2),(1,0))
truth1   = np.expand_dims(np.transpose(sig.rsos(ifft2c_raki(novcc_kspace_truth_raki),-1),(1,0)),axis=0)


rmse_rraki  = np.zeros(len(all_parameters))
rmse_linear = np.zeros(len(all_parameters))

for pp in range(len(all_parameters)):
    rmse_rraki[pp]  = sig.rmse(sig.nor(truth),sig.nor(all_recons_rraki[:,:,pp])) * 100
    # rmse_linear[pp] = sig.rmse(sig.nor(truth),sig.nor(all_recons_linear[:,:,pp])) * 100


# ### Display a particular reconstruciton 

# In[ ]:


pindex = 0

display = sig.nor(np.concatenate((np.expand_dims(all_recons_rraki[:,:,pindex],axis = 0),np.expand_dims(all_est_residual[:,:,pindex],axis=0)*3),axis = 0))
sig.mosaic(display,1,2)


# ### saving
# im_true=sig.nor(truth1)
# im_all_recons=sig.nor(all_recons_rraki)
im_true=truth=truth/np.max(truth)
im_all_recons=all_recons_rraki/np.max(all_recons_rraki)
# In[ ]:
rmse=sig.rmse(np.squeeze(im_true),np.squeeze(im_all_recons))*100
psnr=peak_signal_noise_ratio(np.squeeze(im_true),np.squeeze(im_all_recons))
ssim=structural_similarity(np.squeeze(im_true),np.squeeze(im_all_recons))
print('rmse:',rmse)
print('psnr:',psnr)
print('ssim',ssim)


results = {'truth':          np.squeeze(truth),
           'all_rraki' :     np.squeeze(all_recons_rraki),
           'rraki_rmse':     np.squeeze(rmse_rraki),
           # 'all_linear':     np.squeeze(all_recons_linear),
           'linear_rmse':    np.squeeze(rmse_linear),
           'all_residual':   np.squeeze(all_est_residual),
           'all_parameters': np.squeeze(all_parameters)}

sio.savemat('results/residual_raki_ablation.mat', results, oned_as='row')
plt.subplot(121)
plt.imshow( np.squeeze(im_true),cmap='gray',vmin=0,vmax=1)
plt.title('ori')
plt.subplot(122)
plt.imshow(np.squeeze(im_all_recons),cmap='gray',vmin=0,vmax=1)
plt.title('im_all_recons')
plt.show()

# In[ ]:




