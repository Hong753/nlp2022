# NLP Project 2022

## Preparing the data
Follow https://github.com/tobran/DF-GAN to prepare data. <br>
Unzip birds.zip inside CUB_200_2011, so that all folders are below CUB_200_2011 (get rid of the folder "birds")

## Setting up the environment
```
conda create -n nlp python=3.8 -y
conda activate nlp
conda install spyder jupyter -y # optional
pip3 install torch==1.10 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
conda install nltk pandas scipy
pip3 install yacs==0.1.8
```
