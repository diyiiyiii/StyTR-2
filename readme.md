# StyTr^2 : Image Style Transfer with Transformers（CVPR2022）
*Authors: [Yingying Deng](https://diyiiyiii.github.io/), Fan Tang, XingjiaPan, Weiming Dong, Chongyang Ma, Changsheng Xu*

This paper is proposed to achieve unbiased image style transfer based on the transformer model. We can promote the stylization effect compared with state-of-the-art methods.
This repository is the official implementation of [SyTr^2 : Image Style Transfer with Transformers](https://arxiv.org/abs/2105.14576).

## Results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/Unbiased.png" width="90%" height="90%">
</p>
Compared with some state-of-the-art algorithms, our method has a strong ability to avoid content leakage and has better feature representation ability.  <br>


## Framework
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/network.png" width="100%" height="100%">
</p> 
The overall pipeline of our StyTr^2 framework. We split the content and style images into patches, and use a linear projection to obtain image sequences. Then the content sequences added with CAPE are fed into the content transformer encoder, while the style sequences are fed into the style transformer encoder. Following the two transformer encoders, a multi-layer transformer decoder is adopted to stylize the content sequences according to the style sequences. Finally, we use a progressive upsampling decoder to obtain the stylized images with high-resolution.



## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
```
### Training  
Style dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
content dataset is COCO2014  <br>  
```
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
```
### Reference
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://arxiv.org/abs/2105.14576)<br> 
```
@inproceedings{deng2021stytr2,
      title={StyTr^2: Image Style Transfer with Transformers}, 
      author={Yingying Deng and Fan Tang and Weiming Dong and Chongyang Ma and Xingjia Pan and Lei Wang and Changsheng Xu},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
}
```
