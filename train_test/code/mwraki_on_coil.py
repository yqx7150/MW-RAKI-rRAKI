#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-RAKI imports
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import numpy as np
import numpy.matlib
import time
import os

#-My imports
from utils import signalprocessing as sig
from utils.VCC_signal_creation_nch import circshift,self_floor,VCC_siganal_creation,self_floor1

from  skimage.metrics import peak_signal_noise_ratio
from  skimage.metrics import structural_similarity
# # RAKI definitions 

# In[2]:
os.environ['CUDA_VISIBLE_DEVICES']='1'

def weight_variable(shape,vari_name):                   
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

#### LEANING FUNCTION ####
def learning(ACS_input,target_input,accrate_input,sess):
    
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                  
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])         
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input)) 

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)

    error_norm = tf.norm(input_Target - h_conv3)
    # error_norm = tf.m(input_Target - h_conv3)
    train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1 
    for i in range(MaxIteration):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 50 == 0:
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})    
            print('The',i,'th iteration gives an error',error_now)                             
            
            
        
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),error]  


def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,acc_rate,sess):                
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    return sess.run(h_conv3)           


# # Loading k-space and defining some operators for raki purposes

# In[3]:


fft2c_raki  = lambda x: sig.fft(sig.fft(x,0),1)
ifft2c_raki = lambda x: sig.ifft(sig.ifft(x,0),1)


# In[4]:


image_coils_truth  = sio.loadmat('data/img_grappa_32chan.mat')['IMG']#(188, 236, 32) 图像域的图
# image_coils_truth=sio.loadmat('/home/dt/桌面/TH/th/256DATA1/1.mat')['Img']
# image_coils_truth  = sio.loadmat('/home/dt/桌面/TH/th/256DATA1/1.mat')['Img'];image_coils_truth =image_coils_truth /np.max(image_coils_truth[:])#(188, 236, 32) 图像域的图;


kspace_truth_raki  = fft2c_raki(image_coils_truth)
#数据翻转 创建VCC
#22 把值存起来方便后面比psnr
novcc_kspace_truth_raki=np.copy(kspace_truth_raki)
# novcc_kspace_truth_raki=np.copy(novcc_kspace_truth_raki1[:,:,32:64])
#11
# kspace_truth_raki=VCC_siganal_creation(kspace_truth_raki)

# combine_1_truth_ifft = ifft2c_raki (kspace_truth_raki[:, :, :32])
#
# combine_2_conj_ifft = ifft2c_raki(kspace_truth_raki[:, :, 32:64])
# combine = combine_1_truth_ifft +combine_2_conj_ifft
# plt.figure(1)
# plt.imshow(abs(combine_1_truth_ifft[:,:,2]),cmap='gray')
#
#
# plt.figure(2)
# plt.imshow(abs(combine_2_conj_ifft [:,:,2]),cmap='gray')
#
#
# plt.figure(3)
# plt.imshow(abs(combine[:,:,2]),cmap='gray')
# plt.show()
# assert 0

[M,N,C] = kspace_truth_raki.shape


# # Setting RAKI network parameters 

# In[5]:


#Acquisition/Acceleration Values
Rx     = 1
# all_Ry = [1,2,3,4,5,6]
all_Ry = [6]

acsx     = M
# all_acsy = [20,24,30,36,40]
all_acsy = [40]

#### Network Parameters ####
all_kernel_x_1 = [5]
all_kernel_y_1 = [2]

all_kernel_x_2 = [1]
all_kernel_y_2 = [1]

all_kernel_last_x = [3]
all_kernel_last_y = [2]

all_layer1_channels = [32] 
all_layer2_channels = [8]

MaxIteration = 1000     #Default = 1000
LearningRate = 1e-3 #Default = 3e-3

count_step=0
#kernel_x_1 = 3       #Default = 5
#kernel_y_1 = 2       #Default = 2

#kernel_x_2 = 1       #Default = 1
#kernel_y_2 = 1       #Default = 1

#kernel_last_x = 3    #Default = 3
#kernel_last_y = 2    #Default = 2

#layer1_channels = 32 #Default = 32
#layer2_channels = 8  #Default = 8


# # Trying to copy the RAKI implementation here 

# ## preparing for ablation loop 

# In[6]:


all_kspace_recons = np.zeros((M,N,2*C,len(all_kernel_x_1),len(all_kernel_y_1),len(all_kernel_x_2),                          len(all_kernel_y_2),len(all_kernel_last_x),len(all_kernel_last_y),                          len(all_layer1_channels),len(all_layer2_channels),len(all_Ry),len(all_acsy)),                             dtype = complex)


# In[ ]:


for aa in range(len(all_kernel_x_1)):
    for bb in range(len(all_kernel_y_1)):
        for cc in range(len(all_kernel_x_2)):
            for dd in range(len(all_kernel_y_2)):
                for ee in range(len(all_kernel_last_x)):
                    for ff in range(len(all_kernel_last_y)):
                        for gg in range(len(all_layer1_channels)):
                            for hh in range(len(all_layer2_channels)):
                                for ii in range(len(all_Ry)):
                                    for jj in range(len(all_acsy)):
                                        kernel_x_1 = all_kernel_x_1[aa]       #Default = 5
                                        kernel_y_1 = all_kernel_y_1[bb]       #Default = 2

                                        kernel_x_2 = all_kernel_x_2[cc]       #Default = 1
                                        kernel_y_2 = all_kernel_y_2[dd]       #Default = 1

                                        kernel_last_x = all_kernel_last_x[ee]    #Default = 3
                                        kernel_last_y = all_kernel_last_y[ff]    #Default = 2

                                        layer1_channels = all_layer1_channels[gg] #Default = 32
                                        layer2_channels = all_layer2_channels[hh]  #Default = 8

                                        Ry   = all_Ry[ii]
                                        acsy = all_acsy[jj]#first acs is 20
                                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                        print("R: %d || acs: %d || x1: %d || y1: %d || x2: %d || y2: %d || x3: %d || y3: %d || l1ch: %d || l2ch: %d"                                               % (Ry,acsy,kernel_x_1,kernel_y_1,kernel_x_2,kernel_last_x,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))
                                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                                        #Generate zero-filled ACS
                                        acsregionX = np.arange(M//2 - acsx // 2,M//2 + acsx//2) 
                                        acsregionY = np.arange(N//2 - acsy // 2,N//2 + acsy//2) 

                                        kspace_raki_undersampled_withacs = np.zeros((M,N,C),dtype = complex)#188*236*32
                                        kspace_raki_undersampled_withacs[::Rx,::Ry,:] = kspace_truth_raki[::Rx,::Ry,:]# sample every 5 line only 1 188*236*32
                                        kspace_raki_undersampled_withacs[acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,:] = kspace_truth_raki[acsregionX[0]:acsregionX[acsx-1]+1,acsregionY[0]:acsregionY[acsy-1]+1,:]#after sample copy the ACS regin

                                        kspace = np.copy(kspace_raki_undersampled_withacs)#unsample kspace ACS=[100 20]
                                        no_ACS_flag = 0;
                                        # normalize = 0.015/np.max(abs(kspace[:]))
                                        normalize = 0.15/np.max(abs(kspace[:]))
                                        kspace = np.multiply(kspace,normalize)

                                        kspace_0=np.copy(kspace)
                                        kspace_1=np.copy(kspace)

                                        [m1,n1,no_ch] = np.shape(kspace)
                                        no_inds = 1

                                        kspace_all = np.copy(kspace);
                                        kx = np.transpose(np.int32([(range(1,m1+1))]))  #1 to 188
                                        ky = np.int32([(range(1,n1+1))])#1 to 236
                                        #????
                                        kspace = np.copy(kspace_all)
                                        mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; #y轴大于零的地方236
                                        picks = np.where(mask == 1);  #·[0 5 10 15 ]
                                        kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
                                        kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  

                                        # kspace_NEVER_TOUCH = np.copy(kspace_all)#这几个数据其实是一样的
                                        # kspace_NEVER_TOUCH =VCC_siganal_creation(kspace_all)#这几个数据其实是一样的
                                        kspace_NEVER_TOUCH =np.concatenate((kspace_0,kspace_1),axis=-1)
                                        # kspace_NEVER_TOUCH=np.copy(kspace_NEVER_TOUCH[:,:,no_ch:no_ch*2])

                                        mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; #y轴大小0的地方
                                        picks = np.where(mask == 1);   #得到为1的索引0 5 10 15
                                        d_picks = np.diff(picks,1)  
                                        indic = np.where(d_picks == 1);#得到值为1的索引

                                        mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;#x轴方向
                                        picks_x = np.where(mask_x == 1);##[0 1 2 3 4 ....187]
                                        x_start = picks_x[0][0]#0
                                        x_end = picks_x[0][-1]#187

                                        if np.size(indic)==0:    
                                            no_ACS_flag=1;       
                                            print('No ACS signal in input data, using individual ACS file.')
                                            matfn = 'ACS.mat'   
                                            ACS = sio.loadmat(matfn)
                                            ACS = ACS['ACS']     
                                            [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
                                            ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
                                            ACS_re[:,:,0:no_ch] = np.real(ACS)
                                            ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
                                        else:
                                            no_ACS_flag=0;#ACS是间隔取的当两个的差值为0就表示连续取值
                                            print('ACS signal found in the input data')
                                            indic = indic[1][:]#[22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40] 两维也是最后一维
                                            center_start = picks[0][indic[0]];#108
                                            center_end = picks[0][indic[-1]+1];#128
                                            # ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
                                            ACS_0 = kspace_0[x_start:x_end+1,center_start:center_end+1,:]
                                            ACS_1 = kspace_1[x_start:x_end+1,center_start:center_end+1,:]

                                            # ACS=VCC_siganal_creation(ACS)
                                            # ACS=np.copy(ACS[:,:,no_ch:no_ch*2])
                                            ACS=np.concatenate((ACS_0,ACS_1),axis=-1)

                                            [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)#[188 20 32]
                                            ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])#堆叠[188 20 64 ] 之后填入实部和虚部
                                            ACS_re[:,:,0:no_ch*2] = np.real(ACS)
                                            ACS_re[:,:,no_ch*2:no_ch*4] = np.imag(ACS)

                                        acc_rate = d_picks[0][0]#5
                                        no_channels = ACS_dim_Z*2#64

                                        w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)#[5 2 64 32 64]
                                        w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)#[1 1 32 8 64]
                                        w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32) #[3 2 8 4 64]

                                        b1_flag = 0;
                                        b2_flag = 0;                       
                                        b3_flag = 0;

                                        if (b1_flag == 1):
                                            b1_all = np.zeros([1,1, layer1_channels,no_channels]);
                                        else:
                                            b1 = []

                                        if (b2_flag == 1):
                                            b2_all = np.zeros([1,1, layer2_channels,no_channels])
                                        else:
                                            b2 = []

                                        if (b3_flag == 1):
                                            b3_all = np.zeros([1,1, layer3_channels, no_channels])
                                        else:
                                            b3 = []

                                        target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1); #3
                                        target_x_end = np.int32(ACS_dim_X - target_x_start -1); #184

                                        time_ALL_start = time.time()

                                        [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
                                        ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) #[188 20 64]
                                        ACS = np.float32(ACS)  

                                        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate; #acc_rate=5
                                        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

                                        target_dim_X = target_x_end - target_x_start + 1#182
                                        target_dim_Y = target_y_end - target_y_start + 1#10
                                        target_dim_Z = acc_rate - 1

                                        print('go!')
                                        time_Learn_start = time.time() 

                                        errorSum = 0;
                                        config = tf.ConfigProto()


                                        for ind_c in range(ACS_dim_Z):#64 ACS=·188 20 64「

                                            sess = tf.Session(config=config)
                                            # set target lines
                                            target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])#[1 182 10 4]
                                            print('learning channel #',ind_c+1)#总共64 迭代第一通道
                                            time_channel_start = time.time()

                                            for ind_acc in range(acc_rate-1):
                                                target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
                                                target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
                                                target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

                                            # learning

                                            [w1,w2,w3,error]=learning(ACS,target,acc_rate,sess)
                                            w1_all[:,:,:,:,ind_c] = w1
                                            w2_all[:,:,:,:,ind_c] = w2
                                            w3_all[:,:,:,:,ind_c] = w3                               
                                            time_channel_end = time.time()
                                            print('Time Cost:',time_channel_end-time_channel_start,'s')
                                            print('Norm of Error = ',error)
                                            errorSum = errorSum + error

                                            sess.close()
                                            tf.reset_default_graph()

                                        time_Learn_end = time.time();

                                        print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min')
                                        kspace_recon_all = np.copy(kspace_all)

                                        # kspace_all_1=np.copy(kspace_all_1[:,:,no_ch:no_ch*2])
                                        kspace_recon_all = np.copy(np.concatenate((kspace_0,kspace_1),axis=-1))#得到的容器
                                        kspace_recon_all_nocenter = np.copy(kspace_recon_all)#还是得到一个容器
                                        # kspace_recon_all_nocenter = np.copy(kspace_all)

                                        kspace = np.copy(kspace_all)#欠采数据

                                        over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))#return in 1 not in second [108 109 --127]
                                        kspace_und_0 = kspace_0#还是欠采数据
                                        kspace_und_0[:,over_samp,:] = 0;#欠采之后的数据把中心区域的值变为0

                                        kspace_und_1 = kspace_1  # 还是欠采数据
                                        kspace_und_1[:, over_samp, :] = 0;  # 欠采之后的数据把中心区域的值变为0

                                        kspace_und = np.copy(np.concatenate((kspace_0, kspace_1), axis=-1))  # 得到的容器
                                        # kspace_und=VCC_siganal_creation(kspace_und)#使用VCC 开始的时候是先重建非ACS区域之后把值赋回去
                                        # kspace_und=np.copy(kspace_und[:,:,no_ch:no_ch*2])
                                        # kspace_und=np.concatenate([kspace_und,kspace_und],axis=-1)

                                        [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

                                        kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
                                        kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
                                        kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
                                        kspace_und_re = np.float32(kspace_und_re)
                                        kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
                                        kspace_recon = kspace_und_re

                                        config = tf.ConfigProto()
                                        config.gpu_options.per_process_gpu_memory_fraction = 1/4 ; 

                                        for ind_c in range(0,no_channels):
                                            print('Reconstruting Channel #',ind_c+1)

                                            sess = tf.Session(config=config) 
                                            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                                                init = tf.initialize_all_variables()
                                            else:
                                                init = tf.global_variables_initializer()
                                            sess.run(init)

                                            # grab w and b
                                            w1 = np.float32(w1_all[:,:,:,:,ind_c])
                                            w2 = np.float32(w2_all[:,:,:,:,ind_c])     
                                            w3 = np.float32(w3_all[:,:,:,:,ind_c])

                                            if (b1_flag == 1):
                                                b1 = b1_all[:,:,:,ind_c];
                                            if (b2_flag == 1):
                                                b2 = b2_all[:,:,:,ind_c];
                                            if (b3_flag == 1):
                                                b3 = b3_all[:,:,:,ind_c];                

                                            res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,acc_rate,sess) 
                                            target_x_end_kspace = dim_kspaceUnd_X - target_x_start;

                                            for ind_acc in range(0,acc_rate-1):

                                                target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;             
                                                target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
                                                kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

                                            sess.close()
                                            tf.reset_default_graph()

                                        kspace_recon = np.squeeze(kspace_recon)

                                        kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
                                        kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); 


                                        if no_ACS_flag == 0: 
                                            kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
                                            print('ACS signal has been putted back')
                                        else:
                                            print('No ACS signal is putted into k-space')

                                        kspace_recon_all[:,:,:] = kspace_recon_complex;                                        ##33 最后来这里组装数据  第一步取出实部和虚部分  底可以吗？

                                        #44  组合
                                        combine_1=kspace_recon_all[:,:,0:no_ch]
                                        combine_2=kspace_recon_all[:,:,no_ch:no_ch*2]
                                        combine_2_conj=self_floor1(np.conj(combine_2))
                                        # combine=np.concatenate([combine_1,self_floor1(combine_2)],axis=-1)
                                        combine=np.concatenate([combine_1,combine_2],axis=-1)
                                        # combine= (combine_1+(1e-2)*combine_2_conj)/np.sqrt(2)
                                        # combine= (combine_1+combine_2)/2
                                        # combine= combine_2
                                        # combine=self_floor1(combine)
                                        # combine=self_floor1(combine[:,:,no_ch:no_ch*2])
                                        all_kspace_recons[:, :, :, aa, bb, cc, dd, ee, ff, gg, hh, ii, jj] = combine

                                        # all_kspace_recons[:,:,:,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj] =kspace_recon_all


                                        time_ALL_end = time.time()
                                        print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
                                        print('Error Average in Training is ',errorSum/no_channels)


# #  computing all reconstructions a rmses

# In[ ]:


ablation_dims = [len(all_kernel_x_1),len(all_kernel_y_1),len(all_kernel_x_2),len(all_kernel_y_2),                len(all_kernel_last_x),len(all_kernel_last_y),len(all_layer1_channels),len(all_layer2_channels),                len(all_Ry),len(all_acsy)]
all_rmse   = np.zeros(ablation_dims)
##555 因为只有32 通道
all_recons = sig.rsos(ifft2c_raki(all_kspace_recons),2)

truth      = sig.rsos(ifft2c_raki(novcc_kspace_truth_raki),2)

for aa in range(len(all_kernel_x_1)):
    for bb in range(len(all_kernel_y_1)):
        for cc in range(len(all_kernel_x_2)):
            for dd in range(len(all_kernel_y_2)):
                for ee in range(len(all_kernel_last_x)):
                    for ff in range(len(all_kernel_last_y)):
                        for gg in range(len(all_layer1_channels)):
                            for hh in range(len(all_layer2_channels)):
                                for ii in range(len(all_Ry)):
                                    for jj in range(len(all_acsy)):
                                        all_rmse[aa,bb,cc,dd,ee,ff,gg,hh,ii,jj] = sig.rmse(sig.nor(truth),sig.nor(all_recons[:,:,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj]))


# ## Find parameter set with minimum rmse and then display 

# In[ ]:


cur_rmse = 1e16

for aa in range(len(all_kernel_x_1)):
    for bb in range(len(all_kernel_y_1)):
        for cc in range(len(all_kernel_x_2)):
            for dd in range(len(all_kernel_y_2)):
                for ee in range(len(all_kernel_last_x)):
                    for ff in range(len(all_kernel_last_y)):
                        for gg in range(len(all_layer1_channels)):
                            for hh in range(len(all_layer2_channels)):
                                for ii in range(len(all_Ry)):
                                    for jj in range(len(all_acsy)):
                                        if(all_rmse[aa,bb,cc,dd,ee,ff,gg,hh,ii,jj] < cur_rmse):
                                            best_params = [aa,bb,cc,dd,ee,ff,gg,hh,ii,jj]
                                            cur_rmse = all_rmse[aa,bb,cc,dd,ee,ff,gg,hh,ii,jj]

print("Best Params:")
print("R: %d || acs: %d || x1: %d || y1: %d || x2: %d || y2: %d || x3: %d || y3: %d || l1ch: %d || l2ch: %d" %       (all_Ry[best_params[8]],all_acsy[best_params[9]],all_kernel_x_1[best_params[0]],       all_kernel_y_1[best_params[1]],all_kernel_x_2[best_params[2]],all_kernel_y_2[best_params[3]],        all_kernel_last_x[best_params[4]],all_kernel_last_y[best_params[5]],       all_layer1_channels[best_params[6]],all_layer2_channels[best_params[7]]))
print("Best rmse: %.2f" %(cur_rmse * 100))


# # Quick display 

# In[ ]:


raki = np.expand_dims(np.transpose(all_recons[:,:,best_params[0],best_params[1],best_params[2],best_params[3],best_params[4],                              best_params[5],best_params[6],best_params[7],best_params[8],best_params[9]],(1,0)),axis = 0)

truth   = np.expand_dims(np.transpose(sig.rsos(ifft2c_raki(novcc_kspace_truth_raki),-1),(0,1)),axis=0)
truth1   = np.expand_dims(np.transpose(sig.rsos(ifft2c_raki(novcc_kspace_truth_raki),-1),(1,0)),axis=0)
# display = np.concatenate((sig.nor(truth),sig.nor(raki)),axis = 0)
# sig.mosaic(display,1,2)


# # Assuming I'm only doing ablation over acs size and acceleration, save 

# In[ ]:
im_true=truth=truth/np.max(truth)
# im_true=np.transpose(im_true,(1,0))
im_all_recons=all_recons/np.max(all_recons)
psnr=peak_signal_noise_ratio(np.squeeze(im_true),np.squeeze(im_all_recons))
ssim=structural_similarity(np.squeeze(im_true),np.squeeze(im_all_recons))
print('psnr:',psnr)
print('ssim',ssim)


plt.subplot(121)
plt.imshow( np.squeeze(im_true),cmap='gray',vmin=0,vmax=1)
plt.title('ori')
plt.subplot(122)
plt.imshow(np.squeeze(im_all_recons),cmap='gray',vmin=0,vmax=1)
plt.title('im_all_recons')
plt.show()

print('在通道方向复制原来的信号2份,在通道上用sos迭代1000次看结果')

results = {'truth':   np.squeeze(truth1),
           'all_raki' :   np.squeeze(all_recons),
           'all_raki_rmse': np.squeeze(all_rmse),
           'accelerations': all_Ry,
           'acs_sizes': all_acsy}

sio.savemat('results/raki_ablation.mat', results, oned_as='row')

