# NLP Project 2022

## Preparing the data
Follow https://github.com/tobran/DF-GAN to prepare data. <br>
Unzip birds.zip inside CUB_200_2011, so that all folders are below CUB_200_2011 (get rid of the folder "birds")

## Setting up the environment
```
conda create -n nlp python=3.8 -y
conda activate nlp
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install -c conda-forge nltk pandas scipy
pip3 install yacs==0.1.8
conda install -c conda-forge wandb seaborn
```

## Preprocessing the attributes dataset
Run the codes in ```./NLP2022/parse_attributes.ipynb```. <br>
Make sure the processed attributes go under ```./NLP2022/data/CUB_200_2011/train``` and ```./NLP2022/data/CUB_200_2011/test```.

## Training and Evaluating the Code

Training baseline
```
cd NLP2022
python train.py
```

Training AF-GAN
```
cd NLP2022
python train_attr.py
```

Evaluating
```
cd NLP2022
python eval.py --model_path [MODEL_PATH] --mode [base/attr]
```

To generate images with different attributes or captions, check ```./NLP2022/generate_images.ipynb```

## Reference
The code was heavily inspired by the following paper and code repository.
```
@inproceedings{tao2022df,
  title={DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis},
  author={Tao, Ming and Tang, Hao and Wu, Fei and Jing, Xiao-Yuan and Bao, Bing-Kun and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16515--16525},
  year={2022}
}
```
For calculating the FID and IS score, the following paper and cod repository was referenced.
```
@article{kang2022StudioGAN,
  title   = {{StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis}},
  author  = {MinGuk Kang and Joonghyuk Shin and Jaesik Park},
  journal = {2206.09479 (arXiv)},
  year    = {2022}
}
```