## Kaggle competition
Planet: Understanding the Amazon from Space : https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

## Goal 

Predict a list of tags associated with the image

## Model

We implemented vgg and resnet neural network architectures

K. Simonyan and A. Zisserman <a href="https://arxiv.org/pdf/1409.1556.pdf" class="underline"> Very Deep Convolutional Networks for Large-Scale Image Recognition </a> \
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun   <a href="https://arxiv.org/pdf/1512.03385.pdf" class="underline"> Deep Residual Learning for Image Recognition </a>



## Setup

 1. Download the data from Kaggle
 2. Unzip it
 3. Copy `config.py.example` to `config.py` and edit the path
 4. run the command `python -m make_hdf5.make_hdf5`
 5. CREATE weights and submits folder in root folder (for submit creation)

## HDF5

`python -m make_hdf5.make_hdf5`
`python -m make_hdf5.make_hdf5_tiff`

## Data stream servers

`python -m server_streams.server_train.py --mode=jpeg/tiff`

`python -m server_streams.server_valid.py --mode=jpeg/tiff`

`python -m server_streams.server_submit.py`

## Experiments

`python -m experiments.simple_vgg.train`
`THEANO_FLAGS='floatX=float32,device=gpu' python -m experiments.simple_vgg.train`
`THEANO_FLAGS='floatX=float32,device=gpu' python -m experiments.complex_vgg.train`
`THEANO_FLAGS='floatX=float32,device=gpu' python -m experiments.simple_resnet_152.train`
`THEANO_FLAGS='floatX=float32,device=gpu' python -m experiments.complex_resnet_152.train`

## Tools

Tools or scripts used to explore data. Run using `python -m tools.NAME`

## Submit 

`python -m experiments.simple_vgg.submit`
`THEANO_FLAGS='floatX=float32,device=gpu' python -m experiments.simple_vgg.submit`
