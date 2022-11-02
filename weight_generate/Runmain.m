clc;clear;close all;
Rmax=3e-2;%3e-2
wparam=0.4
weight = WeightMask([188,236],Rmax,wparam);
% weight = hp_weight([256,256],24,10);
%%
path='/media/lqg/KESU/TH/spark_mrm_2021-main/figure3_spark_raki_rraki_comparison/data/weight/'
path1='//media/lqg/KESU/TH/spark_mrm_2021-main/figure3_spark_raki_rraki_comparison/data/'
%%
% path='/home/lqg/桌面/TH/spark_mrm_2021-main/figure3_spark_raki_rraki_comparison/data/256weight/'
% path1='/home/lqg/桌面/TH/spark_mrm_2021-main/figure3_spark_raki_rraki_comparison/data/256DATA1/'
save([path 'weight.mat'],'weight','-v6');
MAX = [];
MAX_ksample=[];

for i = 1:1
%     load(['J:\Dataset\test\train_2coil_SCC\',num2str(i),'.mat']);
    load([path1 'img_grappa_32chan.mat']);
    Img=IMG;
    
%     load([path1 '8.mat']); 
%     load([path1 '111.mat']); 
%     load([path1 '2DMREPPA2.mat']); 
    [M,N,C]=size(Img)
    for j = 1:C
        ori = Img(:,:,j);
        ori = Img(:,:,j)./max(max(abs(Img(:,:,j))));
        kdata = fftshift(fft2(ori));
        k_w = k2wgt(kdata,weight);
        MAX_ksample = [MAX_ksample,max(max(abs(kdata)))];
        MAX = [MAX,max(max(abs(k_w)))];
    end
end


figure(1)
bar(1:C,MAX_ksample);
xlabel('');

ylabel('ģֵ');
figure(2)
bar(1:C,MAX);
xlabel('');

ylabel('g');
figure(3)
mesh(abs(kdata))
% figure(3);
% imshow(log(1+abs(k_w)),[]);
