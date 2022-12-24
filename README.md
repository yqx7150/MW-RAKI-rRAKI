# MW-RAKI-rRAKI
**Paper**: Multi-Weight Respecification of Scan-Specific Learning for Parallel Imaging      
https://arxiv.org/abs/2204.01979     
Available at Magnetic Resonance Imaging, 2023, https://www.sciencedirect.com/science/article/abs/pii/S0730725X2200220X

**Authors**: Hui Tao, Wei Zhang, Haifeng Wang, Shanshan Wang, Dong Liang, Xiaoling Xu*, Qiegen Liu*
Date : 5 Apr 2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

Parallel imaging is widely used in magnetic resonance imaging as an acceleration technology. Traditional linear reconstruction methods in parallel imaging often suffer from noise amplification. Recently, a non-linear robust artificial-neural-network for k-space interpolation (RAKI) exhibits superior noise resil-ience over other linear methods. However, RAKI performs poorly at high acceleration rates and needs a large number of autocalibration signals as the training samples. In order to tackle these issues, we propose a multi-weight method that implements multiple weighting matrices on the under-sampled data, named MW-RAKI. Enforcing multiple weighted matrices on the measurements can effectively reduce the influence of noise and increase the data constraints. Furthermore, we incorporate the strategy of multiple weighting matrixes into a residual version of RAKI, and form MW-rRAKI. Experimental comparisons with the al-ternative methods demonstrated noticeably better reconstruction performances, particularly at high ac-celeration rates. With only 12.5% of the k-space data is available, the PSNR of MW-RAKI and MW-rRAKI is improved by about 3dB and 4dB compared to RAKI and rRAKI, respectively.

## run
```bash
python3 mw_raki.py
python3 mw_rraki.py
```
## Graphical representation
The training and reconstruction flowchart of the proposed MW-RAKI.
 <div align="center"><img src="https://github.com/yqx7150/MW-RAKI-rRAKI/blob/main/train_test/docs/images/flowchart.jpg" width = "800" height = "450">  </div>
Top: The two filters and the auxiliary variable network scheme at the training stage. Bottom: This technique is used for k-space interpolation scan-specific images at the restoration phase.

## Reconstruction results on uniform sampling at acceleration factor R=6 and ACS=40.
<div align="center"><img src="https://github.com/yqx7150/MW-RAKI-rRAKI/blob/main/train_test/docs/images/fig7.jpg"> </div>
Top: Reference, reconstruction by GRAPPA, RAKI, rRAKI, MW-RAKI, and MW-rRAKI. Bottom: The error maps are magnified by 5 times. Green boxes illustrate the zoom in results.

## Other Related Projects
  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
