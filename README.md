# MW-RAKI-rRAKI
**Paper**: Multi-Weight Respecification of Scan-Specific Learning for Parallel Imaging https://arxiv.org/abs/2204.01979

**Authors**: Hui Tao, Wei Zhang,Haifeng Wang,Shanshan Wang,Dong Liang,Xiaoling Xu,Qiegen Liu*,Senior Member,IEEE,Qiegen Liu,Senior Member, IEEE
Date : 5 Apr 2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

Parallel imaging is widely used in magnetic resonance imaging as an acceleration technology. Traditional linear reconstruction methods in parallel imaging often suffer from noise amplification. Recently, a non-linear robust artificial-neural-network for k-space interpolation (RAKI) exhibits superior noise resil-ience over other linear methods. However, RAKI performs poorly at high acceleration rates and needs a large number of autocalibration signals as the training samples. In order to tackle these issues, we propose a multi-weight method that implements multiple weighting matrices on the under-sampled data, named MW-RAKI. Enforcing multiple weighted matrices on the measurements can effectively reduce the influence of noise and increase the data constraints. Furthermore, we incorporate the strategy of multiple weighting matrixes into a residual version of RAKI, and form MW-rRAKI. Experimental comparisons with the al-ternative methods demonstrated noticeably better reconstruction performances, particularly at high ac-celeration rates. With only 12.5% of the k-space data is available, the PSNR of MW-RAKI and MW-rRAKI is improved by about 3dB and 4dB compared to RAKI and rRAKI, respectively.

## run
```bash
python3 mw_raki.py
```
## Graphical representation
The training and reconstruction flowchart of the proposed MW-RAKI.
 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig6.png" width = "400" height = "450">  </div>
Top: The two filters and the auxiliary variable network scheme at the training stage. Bottom: This technique is used for k-space interpolation scan-specific images at the restoration phase.
