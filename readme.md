# StyTr^2 : Unbiased Image Style Transfer with Transformers
Yingying Deng, Fan Tang, XingjiaPan, Weiming Dong, Chongyang Ma, Changsheng Xu <br>

## results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/StyTR-2/blob/main/Figure/Unbiased.png" width="50%" height="50%">
</p>
Stylized result using Claude Monet's painting as style reference. Compared with some state-of-the-art algorithms, our result can preserve detailed content structures and maintain vivid style patterns.  <br>


## Framework
<p align="center">
<img src="https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/blob/master/framework/framework1.png" width="80%" height="80%">
</p> 
System overview. For the purpose of arbitrary style transfer, we propose a feed-forward network, which contains an encoder-decoder architecture and a multi-adaptation module.


<p align="center">
<img src="https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/blob/master/framework/SACA1.png" width="80%" height="80%">
</p> 
The multi-adaptation module is divided into three parts: position-wise content SA module, channel-wise style SA module, and CA module.  <br>


## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [decoder, MA_module](https://drive.google.com/file/d/1JaFfo6VB5NG0y69YfmB6WWjODQNdymWt/view?usp=sharing)   <br> 
Please download them and put them into the floder  ./model/  <br> 
```
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
```
### Training  
Traing set is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
Testing set is COCO2014  <br>  
```
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 4
```
### Reference
If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://arxiv.org/abs/2005.13219)<br> 
```
@inproceedings{deng:2020:arbitrary,
  title={Arbitrary Style Transfer via Multi-Adaptation Network},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Sun, Wen, and Huang, Feiyue and Xu, Changsheng},
  booktitle={Acm International Conference on Multimedia},
  year={2020},
 publisher = {ACM},
}
```
